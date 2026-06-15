"""Text-to-motion generation with LLM prompt expansion and adaptive sampling.

Main entry point for the MoConVQ inference pipeline:
  prompt → LLM expand + strategy → T5 encode → MoConGPT sample → physics decode → BVH + GIF

Usage:
  python Generation/generate.py          # batch test all prompts
  python Generation/generate.py --prompt "A person kicks."  # single prompt
"""

import sys
import os

# Ensure MoConVQ/ and project root are on the Python path
# DingHao/Generation/generate.py -> DingHao/ -> project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MOCONVQ_ROOT = os.path.join(_PROJECT_ROOT, "MoConVQ")
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, _MOCONVQ_ROOT)

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from torch.distributions import Categorical

from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
import MoConVQCore.Utils.pytorch_utils as ptu

from DingHao.Generation.sampling_strategies import SamplingConfig, get_strategy, PRESETS, DEFAULT_STRATEGY
from DingHao.Generation.prompt_engine import PromptEngine


# ── T5 text encoder ──────────────────────────────────────────────────

_bert = None
_bert_tokenizer = None


def _init_t5():
    global _bert, _bert_tokenizer
    if _bert is not None:
        return
    from transformers import T5Tokenizer, T5EncoderModel
    t5_path = os.path.join(_MOCONVQ_ROOT, 't5-large')
    _bert_tokenizer = T5Tokenizer.from_pretrained(t5_path)
    _bert = T5EncoderModel.from_pretrained(t5_path).to(ptu.device)
    _bert.eval()


def text2bert(text):
    encoded_input = _bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encoded_input = {key: value.to(ptu.device) for key, value in encoded_input.items()}
    with torch.no_grad():
        output = _bert(**encoded_input)
    return output.last_hidden_state, ~encoded_input['attention_mask'].bool()


# ── Model config ─────────────────────────────────────────────────────

class gpt_config:
    def __init__(self):
        self.num_vq = 512
        self.embed_dim = 768
        self.clip_dim = 512
        self.block_size = 52
        self.num_layers = 9
        self.n_head = 8
        self.drop_out_rate = 0.1
        self.fc_rate = 2


# ── Trainer (generation logic) ───────────────────────────────────────

class Trainer:
    def __init__(self, agent, device_list=None):
        embed_torch = [
            torch.cat([bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])], dim=0)
            for bottle_neck in agent.posterior.bottle_neck_list
        ]
        gpt = Text2Motion_Transformer(**vars(gpt_config()), embeddings=embed_torch).to(ptu.device)
        if device_list is None:
            device_list = [ptu.device.index if hasattr(ptu.device, 'index') else 0]
        self.gpt = nn.DataParallel(gpt, device_ids=device_list)
        self.name = 'text2motion'

    def evaluate_generate(self, text, idx, out_dir, agent, env,
                          sampling_config=None, use_viewer=False,
                          record_viewer=False, **kargs):
        """Generate motion from text and save BVH + GIF.

        Parameters
        ----------
        sampling_config : SamplingConfig, optional
            Temperature / top_k / top_p settings. Uses moderate (default) if None.
        use_viewer : bool
            Launch ODE OpenGL 3D viewer during physics simulation.
        record_viewer : bool
            Record 3D view as GIF instead of holding window open.
        """
        self._record_viewer = record_viewer
        if sampling_config is None:
            sampling_config = get_strategy(DEFAULT_STRATEGY)

        if 'bert_feature' not in kargs:
            bert_feature, bert_mask = text2bert(text)
        else:
            bert_feature = kargs['bert_feature']
            bert_mask = kargs['bert_mask']

        try:
            gpt = self.gpt.module
        except AttributeError:
            gpt = self.gpt

        clip_feature = torch.zeros((1, 512)).to(ptu.device)
        cur_embedding, _ = gpt.sample(
            clip_feature, bert_feature, bert_mask,
            temperature=sampling_config.temperature,
            top_k=sampling_config.top_k,
            top_p=sampling_config.top_p,
        )
        dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

        import VclSimuBackend
        CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
        saver = CharacterToBVH(agent.env.sim_character, 120)
        saver.bvh_hierarchy_no_root()

        observation, info = agent.env.reset(0)

        # OpenGL 3D viewer (optional) — created AFTER reset so camera
        # can target the character's actual spawn position.
        viewer = None
        if use_viewer:
            from DingHao.Visualization.ode_viewer import ODEViewer
            root_pos = agent.env.sim_character.root_body.PositionNumpy
            viewer = ODEViewer(agent.env, center=root_pos)
            viewer.start()
            if getattr(self, '_record_viewer', False):
                viewer.start_recording()

        num_frames = dconv.shape[1]

        # With viewer: loop the motion so character doesn't stand idle
        _pass = 0
        while True:
            for i in range(num_frames):
                obs = observation['observation']
                action, info = agent.act_tracking(
                    obs_history=[obs.reshape(1, 323)],
                    target_latent=dconv[:, i],
                )
                action = ptu.to_numpy(action).flatten()
                for _ in range(6):
                    if _pass == 0:  # record BVH only on first pass
                        saver.append_no_root_to_buffer()
                    if _ == 0:
                        step_generator = agent.env.step_core(action, using_yield=True)
                    info = next(step_generator)
                try:
                    info_ = next(step_generator)
                except StopIteration as e:
                    info_ = e.value
                new_observation, rwd, done, info = info_
                observation = new_observation

                if viewer:
                    viewer.tick()

            _pass += 1
            if self._record_viewer and _pass == 1:
                # Stop recording after first pass, save 3D GIF
                record_path = os.path.join(out_dir, f'evaluate_gpt{idx}_3d.gif')
                viewer.save_gif(record_path, fps=20, max_frames=200)
                viewer = None  # don't hold/loop, continue to BVH/GIF output
                break
            if not viewer:
                break
            # Reset to loop start for continuous playback
            observation, info = agent.env.reset(0)

        if viewer:
            if getattr(self, '_record_viewer', False):
                record_path = os.path.join(out_dir, f'evaluate_gpt{idx}_3d.gif')
                viewer.save_gif(record_path, fps=20)
            else:
                viewer.hold()

        os.makedirs(out_dir, exist_ok=True)
        bvh_path = os.path.join(out_dir, f'evaluate_gpt{idx}.bvh')
        saver.to_file(bvh_path)

        try:
            from DingHao.Visualization.bvh_visualizer import visualize_bvh
            gif_path = os.path.join(out_dir, f'evaluate_gpt{idx}.gif')
            print(f'  Rendering skeleton visualization ...')
            visualize_bvh(bvh_path, gif_path, output_fps=30)
        except Exception as e:
            print(f'  Single-angle viz skipped (non-fatal): {e}')

        try:
            from DingHao.Visualization.multi_angle_viewer import visualize_multi_angle
            multi_path = os.path.join(out_dir, f'evaluate_gpt{idx}_multi.gif')
            print(f'  Rendering multi-angle 3D view ...')
            visualize_multi_angle(bvh_path, multi_path, output_fps=15, dpi=100)
        except Exception as e:
            print(f'  Multi-angle viz skipped (non-fatal): {e}')

        return bvh_path


# ── Main ─────────────────────────────────────────────────────────────

def main():
    agent, env, trainer = _setup()
    engine = PromptEngine()
    out_dir = os.path.join(_PROJECT_ROOT, 'out', 'conditional')

    prompts = [
        "A person is jogging forward.",
        "A person kicks with their right leg.",
        "A person throws a punch with their right hand.",
        "A person jumps up and down.",
        "A person walks forward and then turns around.",
        # Zero-shot prompts
        "一个人高兴地跳舞。",
        "A person sneaks quietly like a ninja.",
        "A person swims freestyle.",
        "A person acts like they're surprised by something.",
        "A person does a ballet spin.",
        "一个人打太极拳。",
        "A person tiptoes to avoid making noise.",
    ]

    for idx, text in enumerate(prompts):
        print(f'\n{"=" * 60}')
        print(f'[{idx}] Original:  "{text}"')

        # LLM expand + strategy selection
        result = engine.process(text)
        expanded = result["expanded_prompt"]
        strategy_name = result["strategy"]
        strategy = get_strategy(strategy_name)

        print(f'    Expanded:  "{expanded}"')
        print(f'    Strategy:  {strategy_name}  '
              f'(T={strategy.temperature:.1f}, '
              f'top_k={strategy.top_k if strategy.top_k else "-"}, '
              f'top_p={strategy.top_p if strategy.top_p is not None else "-"})')
        print(f'    Reason:    {result.get("reason", "N/A")}')

        bert_feature, bert_mask = text2bert(expanded)
        trainer.evaluate_generate(
            expanded, idx, out_dir,
            agent=agent, env=env,
            sampling_config=strategy,
            bert_feature=bert_feature, bert_mask=bert_mask,
        )

    print(f'\n{"=" * 60}')
    print(f'Done. Outputs in: {out_dir}')


def _setup(device=0):
    """Load models and return (agent, env, trainer). Call once at startup."""
    # build_agent uses relative paths (Data/Parameters/bigdata.yml) — chdir to MoConVQ/
    _saved_cwd = os.getcwd()
    _saved_argv = sys.argv
    sys.argv = [sys.argv[0]]
    os.chdir(_MOCONVQ_ROOT)
    try:
        from Script.moconvq_builder import build_agent
        agent, env = build_agent(gpu=device)
    finally:
        os.chdir(_saved_cwd)
        sys.argv = _saved_argv

    ptu.init_gpu(True, gpu_id=device)
    _init_t5()

    agent.simple_load(os.path.join(_MOCONVQ_ROOT, 'moconvq_base.data'), strict=True)
    agent.eval()

    trainer = Trainer(agent, device_list=[device])
    trainer.gpt.load_state_dict(
        torch.load(os.path.join(_MOCONVQ_ROOT, 'text_generation_GPT.pth'),
                   map_location=ptu.device)
    )
    trainer.gpt = trainer.gpt.eval()

    return agent, env, trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Text-to-motion generation')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Single prompt (default: batch test all prompts)')
    parser.add_argument('--viewer', action='store_true',
                        help='Launch ODE OpenGL 3D viewer during physics simulation')
    parser.add_argument('--record', action='store_true',
                        help='Record OpenGL 3D view as GIF (implies --viewer)')
    args, _ = parser.parse_known_args()

    if args.prompt:
        agent, env, trainer = _setup()
        engine = PromptEngine()
        result = engine.process(args.prompt)
        strategy = get_strategy(result["strategy"])
        print(f'Expanded:  "{result["expanded_prompt"]}"')
        print(f'Strategy:  {result["strategy"]} '
              f'(T={strategy.temperature:.1f}, '
              f'top_k={strategy.top_k if strategy.top_k else "-"}, '
              f'top_p={strategy.top_p if strategy.top_p is not None else "-"})')

        if args.record:
            args.viewer = True

        out_dir = os.path.join(_PROJECT_ROOT, 'out', 'conditional')
        os.makedirs(out_dir, exist_ok=True)
        bert_feature, bert_mask = text2bert(result["expanded_prompt"])
        trainer.evaluate_generate(
            result["expanded_prompt"], 0, out_dir,
            agent=agent, env=env,
            sampling_config=strategy,
            bert_feature=bert_feature, bert_mask=bert_mask,
            use_viewer=args.viewer,
            record_viewer=args.record,
        )
    else:
        main()
