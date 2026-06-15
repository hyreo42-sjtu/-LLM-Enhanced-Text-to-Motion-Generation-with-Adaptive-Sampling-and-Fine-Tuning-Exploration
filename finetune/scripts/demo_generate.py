"""Generate motions using the fine-tuned model for demo texts."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ModifyODESrc'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'diff-quaternion', 'TorchRotation'))

import numpy as np, torch, time
import MoConVQCore.Utils.pytorch_utils as ptu

# Load models once
old_argv = sys.argv[:]
sys.argv = [sys.argv[0], '--experiment_name', 'demo']
from Script.moconvq_builder import build_agent
agent, env = build_agent(gpu=0)
sys.argv = old_argv
ptu.init_gpu(True, gpu_id=0)
agent.simple_load('moconvq_base.data', strict=True); agent.eval()
torch.set_grad_enabled(False)

from transformers import T5Tokenizer, T5EncoderModel
tok = T5Tokenizer.from_pretrained('t5-large', resume_download=True)
t5 = T5EncoderModel.from_pretrained('t5-large', resume_download=True).to(ptu.device).eval()

from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer
class Cfg:
    def __init__(s): s.num_vq=512; s.embed_dim=768; s.clip_dim=512; s.block_size=52; s.num_layers=9; s.n_head=8; s.drop_out_rate=0.1; s.fc_rate=2

embed_torch = [torch.cat([bn.embedding, torch.zeros_like(bn.embedding[:2])], dim=0) for bn in agent.posterior.bottle_neck_list]
embed_torch = [e.to(ptu.device) for e in embed_torch]

# Load fine-tuned model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'best_model.pth')
gpt = Text2Motion_Transformer(**vars(Cfg()), embeddings=embed_torch).to(ptu.device)
gpt.load_state_dict(torch.load(MODEL_PATH, map_location=ptu.device)['model_state_dict'], strict=False)
gpt.eval()

import VclSimuBackend; CharacterToBVH = VclSimuBackend.ODESim.CharacterTOBVH

def generate(text, out_path):
    enc = tok(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
    enc = {k: v.to(ptu.device) for k, v in enc.items()}
    with torch.no_grad():
        bf, bm = t5(**enc).last_hidden_state, ~enc['attention_mask'].bool()
        cur_emb, _ = gpt.sample(torch.zeros((1, 512), device=ptu.device), bf, bm)
    dconv = agent.posterior.decoder.decode_dynamic(cur_emb)
    saver = CharacterToBVH(env.sim_character, 120); saver.bvh_hierarchy_no_root()
    obs, info = env.reset(0)
    for j in range(min(dconv.shape[1], 300)):
        action, info = agent.act_tracking(obs_history=[obs['observation'].reshape(1,323)], target_latent=dconv[:,j])
        action = ptu.to_numpy(action).flatten()
        for k in range(6):
            saver.append_no_root_to_buffer()
            if k == 0: sg = env.step_core(action, using_yield=True)
            info = next(sg)
        try: info_ = next(sg)
        except StopIteration as e: info_ = e.value
        obs = info_[0]
    saver.to_file(out_path)
    return dconv.shape[1]

if __name__ == '__main__':
    import json
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demo', 'texts.json')) as f:
        texts = json.load(f)
    os.makedirs('demo_out', exist_ok=True)
    for name, text in texts.items():
        bvh = f'demo_out/{name}_frozen.bvh'
        print(f"Generating: {name}...", end=' ', flush=True)
        t0 = time.time()
        nf = generate(text, bvh)
        print(f"{nf}f, {time.time()-t0:.1f}s")
    print("Done!")
