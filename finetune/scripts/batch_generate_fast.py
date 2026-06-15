#!/usr/bin/env python3
"""
Efficient batch text-to-motion generation.
Loads models once, then generates motions for all prompts sequentially.
"""

import sys
import os
import json
import time
import argparse

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib/python3.8/dist-packages/torch/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'ModifyODESrc'))
sys.path.insert(0, os.path.join(BASE_DIR, 'diff-quaternion', 'TorchRotation'))

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import h5py
from torch.utils.data import DataLoader
import MoConVQCore.Utils.pytorch_utils as ptu


def text2bert(text, bert, bert_tokenizer):
    encoded_input = bert_tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    encoded_input = {key: value.to(ptu.device) for key, value in encoded_input.items()}
    with torch.no_grad():
        output = bert(**encoded_input)
    return output.last_hidden_state, ~encoded_input['attention_mask'].bool()


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


def generate_one_motion(agent, env, trainer, bert, bert_tokenizer, text, output_path):
    """Generate a single motion and save to output_path."""
    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)

    bert_feature, bert_mask = text2bert(text, bert, bert_tokenizer)

    # Generate
    try:
        gpt = trainer.gpt.module
    except:
        gpt = trainer.gpt

    clip_feature = torch.zeros((1, 512)).to(ptu.device)
    cur_embedding, _ = gpt.sample(clip_feature, bert_feature, bert_mask)

    dconv = agent.posterior.decoder.decode_dynamic(cur_embedding)

    import VclSimuBackend
    CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH
    saver = CharacterToBVH(agent.env.sim_character, 120)
    saver.bvh_hierarchy_no_root()

    observation, info = agent.env.reset(0)

    for j in range(dconv.shape[1]):
        obs = observation['observation']
        action, info = agent.act_tracking(
            obs_history=[obs.reshape(1, 323)],
            target_latent=dconv[:, j],
        )
        action = ptu.to_numpy(action).flatten()
        for i in range(6):
            saver.append_no_root_to_buffer()
            if i == 0:
                step_generator = agent.env.step_core(action, using_yield=True)
            info = next(step_generator)

        try:
            info_ = next(step_generator)
        except StopIteration as e:
            info_ = e.value
        new_observation, rwd, done, info = info_
        observation = new_observation

    saver.to_file(output_path)
    return dconv.shape[1]  # Return number of frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default=None, help='Only process specific category')
    parser.add_argument('--prompts', type=str, nargs='*', default=None, help='Specific prompt keys')
    parser.add_argument('--prompts-file', type=str, default=None, help='JSON file with {key: text} prompts')
    parser.add_argument('--start', type=int, default=0, help='Start index in prompt list')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for BVH files')
    args = parser.parse_args()

    DATASET_DIR = os.path.join(BASE_DIR, 'out', 'finetune_dataset')
    BVH_DIR = args.output_dir if args.output_dir else os.path.join(DATASET_DIR, 'bvh')
    os.makedirs(BVH_DIR, exist_ok=True)

    # Determine prompts to process
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            MOTION_PROMPTS = json.load(f)
        CATEGORIES = {}
        prompt_keys = list(MOTION_PROMPTS.keys()) if not args.prompts else args.prompts
    elif args.prompts:
        from Script.batch_generate_motions import MOTION_PROMPTS
        prompt_keys = args.prompts
    elif args.category:
        from Script.batch_generate_motions import MOTION_PROMPTS, CATEGORIES
        prompt_keys = CATEGORIES.get(args.category, [])
    else:
        from Script.batch_generate_motions import MOTION_PROMPTS
        prompt_keys = list(MOTION_PROMPTS.keys())

    prompt_keys = prompt_keys[args.start:]
    print(f"Will generate {len(prompt_keys)} motions")
    print(f"Output directory: {BVH_DIR}")

    # ── Load models once ──
    print("\n[1/4] Building agent and environment...")
    from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
    from Script.moconvq_builder import build_agent

    # Save and modify argv for moconvq_builder
    old_argv = sys.argv[:]
    sys.argv = [sys.argv[0], '--experiment_name', 'batch_gen']
    agent, env = build_agent(gpu=args.gpu)
    sys.argv = old_argv

    ptu.init_gpu(True, gpu_id=args.gpu)

    print("[2/4] Loading T5 text encoder...")
    from transformers import T5Tokenizer, T5EncoderModel
    bert_tokenizer = T5Tokenizer.from_pretrained('t5-large', resume_download=True)
    bert = T5EncoderModel.from_pretrained('t5-large', resume_download=True).to(ptu.device)
    bert.eval()

    print("[3/4] Loading MoConVQ model...")
    agent.simple_load('moconvq_base.data', strict=True)
    agent.eval()

    print("[4/4] Loading GPT model...")
    embed_torch = [
        torch.cat([bottle_neck.embedding, torch.zeros_like(bottle_neck.embedding[:2])], dim=0)
        for bottle_neck in agent.posterior.bottle_neck_list
    ]
    gpt = Text2Motion_Transformer(**vars(gpt_config()), embeddings=embed_torch).to(ptu.device)
    trainer = type('Trainer', (), {'gpt': gpt})()
    state_dict = torch.load('text_generation_GPT.pth', map_location=ptu.device)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    trainer.gpt.load_state_dict(state_dict)
    trainer.gpt = trainer.gpt.eval()

    print("\nAll models loaded. Starting generation...\n")

    # ── Generate motions ──
    results = {}
    for i, key in enumerate(prompt_keys):
        text = MOTION_PROMPTS[key]
        bvh_path = os.path.join(BVH_DIR, f'{key}.bvh')

        if os.path.exists(bvh_path):
            print(f"[{i+1}/{len(prompt_keys)}] SKIP {key} (exists)")
            results[key] = 'skipped'
            continue

        print(f"[{i+1}/{len(prompt_keys)}] {key}: '{text[:80]}...' ", end='', flush=True)
        t0 = time.time()
        try:
            n_frames = generate_one_motion(agent, env, trainer, bert, bert_tokenizer, text, bvh_path)
            elapsed = time.time() - t0
            print(f"✓ {n_frames} frames, {elapsed:.1f}s")
            results[key] = 'success'
        except Exception as e:
            elapsed = time.time() - t0
            print(f"✗ {str(e)[:100]}, {elapsed:.1f}s")
            results[key] = f'error: {str(e)[:200]}'

    # Save results
    with open(os.path.join(DATASET_DIR, 'generation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    success = sum(1 for v in results.values() if v == 'success' or v == 'skipped')
    failed = sum(1 for v in results.values() if v not in ('success', 'skipped'))
    print(f"\n{'='*60}")
    print(f"Done! Success: {success}, Failed: {failed}, Total: {len(prompt_keys)}")
    print(f"Outputs: {BVH_DIR}/")


if __name__ == '__main__':
    main()
