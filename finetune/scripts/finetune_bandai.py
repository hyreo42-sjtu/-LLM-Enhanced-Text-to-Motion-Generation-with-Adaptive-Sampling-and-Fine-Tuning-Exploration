#!/usr/bin/env python3
"""
Fine-tune MoConVQ Text2Motion_Transformer on Bandai Namco style-labeled dataset.

Usage: python Script/finetune_bandai.py [--epochs 50] [--lr 1e-5] [--plot]
"""

import sys, os, json, argparse, time, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'ModifyODESrc'))
sys.path.insert(0, os.path.join(BASE_DIR, 'diff-quaternion', 'TorchRotation'))

import MoConVQCore.Utils.pytorch_utils as ptu
from MoConVQCore.Model.cross_trans_ori_fixsum import Text2Motion_Transformer


class GPTConfig:
    def __init__(self):
        self.num_vq = 512; self.embed_dim = 768; self.clip_dim = 512
        self.block_size = 52; self.num_layers = 9; self.n_head = 8
        self.drop_out_rate = 0.1; self.fc_rate = 2


class BandaiTextMotionDataset(Dataset):
    def __init__(self, dataset_json, tokens_npz, split='train', max_seq_len=50, max_depth=4,
                 bert_tokenizer=None, bert_model=None):
        self.max_seq_len = max_seq_len; self.max_depth = max_depth

        with open(dataset_json) as f:
            ds = json.load(f)
        self.entries = ds[split]
        print(f"BandaiDataset ({split}): {len(self.entries)} entries")

        # Load all tokens
        self.tokens = dict(np.load(tokens_npz, allow_pickle=True))
        # Pre-compute text embeddings
        self.text_embeddings = {}; self.text_masks = {}
        if bert_tokenizer and bert_model:
            self._precompute_texts(bert_tokenizer, bert_model)
        # Build sliding windows
        self._build_windows()

    def _precompute_texts(self, tok, model):
        texts = list(set(e['text'] for e in self.entries))
        for i, text in enumerate(texts):
            with torch.no_grad():
                enc = tok(text, return_tensors='pt', padding=True, truncation=True, max_length=256)
                enc = {k: v.to(ptu.device) for k, v in enc.items()}
                out = model(**enc)
                self.text_embeddings[text] = out.last_hidden_state.cpu()
                self.text_masks[text] = (~enc['attention_mask'].bool()).cpu()
            if (i+1) % 500 == 0:
                print(f"  Precomputed {i+1}/{len(texts)} texts")

    def _build_windows(self):
        self.windows = []
        for entry in self.entries:
            key = entry['token_key']; text = entry['text']
            if key not in self.tokens: continue
            tokens = self.tokens[key]  # (frame_len, n_layers)
            if tokens.ndim != 2: continue
            seq_len = tokens.shape[0]
            # Use a smaller effective max_seq_len for short motions
            effective_max = min(self.max_seq_len, seq_len)
            if effective_max >= 4:
                stride = max(1, effective_max // 2)
                for start in range(0, max(1, seq_len - effective_max + 1), stride):
                    self.windows.append((key, text, start))
            elif seq_len >= 3:
                self.windows.append((key, text, 0))
        print(f"  Windows: {len(self.windows)} (avg {len(self.windows)/max(1,len(self.entries)):.1f}/entry)")

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        key, text, start = self.windows[idx]
        tokens = self.tokens[key]  # (seq_len, n_layers)
        end = min(start + self.max_seq_len + 1, tokens.shape[0])
        window = tokens[start:end]
        # Pad
        target_len = self.max_seq_len + 1
        if window.shape[0] < target_len:
            pad = np.full((target_len - window.shape[0], tokens.shape[1]), 512, dtype=np.int64)
            window = np.concatenate([window, pad], axis=0)
        # Pad depth to max_depth (some files have fewer layers)
        d = self.max_depth
        if window.shape[1] < d:
            pad_d = np.full((window.shape[0], d - window.shape[1]), 512, dtype=np.int64)
            window = np.concatenate([window[:, :window.shape[1]], pad_d], axis=1)
        else:
            window = window[:, :d]
        input_t = window[:self.max_seq_len]; target_t = window[1:self.max_seq_len+1]
        # Text embedding
        if text in self.text_embeddings:
            bf = self.text_embeddings[text].squeeze(0); bm = self.text_masks[text].squeeze(0)
        else:
            bf = torch.zeros(1, 1024); bm = torch.ones(1, dtype=torch.bool)
        return {'input_tokens': torch.from_numpy(input_t).long(),
                'target_tokens': torch.from_numpy(target_t).long(),
                'bert_feature': bf, 'bert_mask': bm}


def collate_fn(batch):
    input_tokens = torch.stack([b['input_tokens'] for b in batch])
    target_tokens = torch.stack([b['target_tokens'] for b in batch])
    max_len = max(b['bert_feature'].shape[0] for b in batch)
    dim = batch[0]['bert_feature'].shape[-1]; bs = len(batch)
    bf = torch.zeros(bs, max_len, dim); bm = torch.ones(bs, max_len, dtype=torch.bool)
    for i, b in enumerate(batch):
        l = b['bert_feature'].shape[0]; bf[i,:l] = b['bert_feature']; bm[i,:l] = b['bert_mask']
    return {'input_tokens': input_tokens, 'target_tokens': target_tokens,
            'bert_feature': bf, 'bert_mask': bm}


def run_forward(model, input_tokens, bert_feature, bert_mask, clip_feature=None):
    device = input_tokens.device
    b, t, d = input_tokens.shape
    if clip_feature is None:
        clip_feature = torch.zeros((b, 512), device=device)
    latents = torch.zeros((b, t, 768), device=device)
    for depth in range(d):
        tok_emb = model.trans_base.tok_emb[depth](input_tokens[:,:,depth])
        latents = latents + tok_emb if depth > 0 else tok_emb
    temporal_feat = model.trans_temporal(latents, clip_feature, bert_feature, bert_mask)
    temporal_feat = temporal_feat[:, -t:, :]
    base_feat = model.trans_base(input_tokens.reshape(b*t, d), temporal_feat.reshape(b*t, 768))
    logits = model.trans_head(base_feat).view(b, t, d+1, -1)
    pred_logits = logits[:,:,1:,:].reshape(-1, logits.shape[-1])
    targets_arg = None  # handled outside
    return pred_logits


def train_epoch(model, loader, optimizer, device, epoch):
    model.train(); total_loss = 0.0; total_tokens = 0
    for bi, batch in enumerate(loader):
        it = batch['input_tokens'].to(device); tt = batch['target_tokens'].to(device)
        bf = batch['bert_feature'].to(device); bm = batch['bert_mask'].to(device)
        cf = torch.zeros((it.shape[0], 512), device=device)

        b, t, d = it.shape
        latents = torch.zeros((b, t, 768), device=device)
        for depth in range(d):
            te = model.trans_base.tok_emb[depth](it[:,:,depth])
            latents = latents + te if depth > 0 else te

        temporal_feat = model.trans_temporal(latents, cf, bf, bm)
        temporal_feat = temporal_feat[:, -t:, :]
        base_feat = model.trans_base(it.reshape(b*t, d), temporal_feat.reshape(b*t, 768))
        logits = model.trans_head(base_feat).view(b, t, d+1, -1)

        pred = logits[:,:,1:,:].reshape(-1, logits.shape[-1])
        targets = tt.reshape(-1)
        mask = (targets < 512); pred = pred[mask]; targets = targets[mask]
        if targets.numel() == 0: continue

        loss = F.cross_entropy(pred, targets)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()

        total_loss += loss.item() * targets.numel(); total_tokens += targets.numel()
        if bi % 20 == 0:
            print(f"  E{epoch} B{bi}/{len(loader)} loss={loss.item():.4f} ppl={torch.exp(loss).item():.1f}")
    return total_loss/max(total_tokens,1), np.exp(total_loss/max(total_tokens,1))


@torch.no_grad()
def validate(model, loader, device):
    model.eval(); total_loss = 0.0; total_tokens = 0
    for batch in loader:
        it = batch['input_tokens'].to(device); tt = batch['target_tokens'].to(device)
        bf = batch['bert_feature'].to(device); bm = batch['bert_mask'].to(device)
        cf = torch.zeros((it.shape[0], 512), device=device)

        b, t, d = it.shape
        latents = torch.zeros((b, t, 768), device=device)
        for depth in range(d):
            te = model.trans_base.tok_emb[depth](it[:,:,depth])
            latents = latents + te if depth > 0 else te

        temporal_feat = model.trans_temporal(latents, cf, bf, bm)
        temporal_feat = temporal_feat[:, -t:, :]
        base_feat = model.trans_base(it.reshape(b*t, d), temporal_feat.reshape(b*t, 768))
        logits = model.trans_head(base_feat).view(b, t, d+1, -1)

        pred = logits[:,:,1:,:].reshape(-1, logits.shape[-1])
        targets = tt.reshape(-1)
        mask = (targets < 512); pred = pred[mask]; targets = targets[mask]
        if targets.numel() == 0: continue
        loss = F.cross_entropy(pred, targets)
        total_loss += loss.item() * targets.numel(); total_tokens += targets.numel()
    return total_loss/max(total_tokens,1), np.exp(total_loss/max(total_tokens,1))


def plot_loss(history, save_path):
    epochs = [h['epoch']+1 for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_loss, 'b-o', label='Train Loss', markersize=6)
    ax1.plot(epochs, val_loss, 'r-s', label='Val Loss', markersize=6)
    best_epoch = epochs[np.argmin(val_loss)]
    best_val = min(val_loss)
    ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                label=f'Best: epoch {best_epoch} (val={best_val:.2f})')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training and Validation Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    train_ppl = [h['train_ppl'] for h in history]
    val_ppl = [h['val_ppl'] for h in history]
    ax2.plot(epochs, train_ppl, 'b-o', label='Train PPL', markersize=6)
    ax2.plot(epochs, val_ppl, 'r-s', label='Val PPL', markersize=6)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Loss curve saved to: {save_path}")

    # Also print summary
    print(f"\n=== Training Summary ===")
    print(f"Best epoch: {best_epoch}, Best val loss: {best_val:.4f}")
    print(f"Final train loss: {train_loss[-1]:.4f}, Final val loss: {val_loss[-1]:.4f}")
    baseline_random = np.log(513)
    print(f"Random baseline loss: {baseline_random:.4f} (ln(513))")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50); parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=50); parser.add_argument('--max_depth', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dataset_json', type=str, default='out/bandai_dataset/text_motion_dataset.json')
    parser.add_argument('--tokens_npz', type=str, default='out/bandai_dataset/tokens.npz')
    parser.add_argument('--checkpoint_dir', type=str, default='out/bandai_dataset/checkpoints')
    parser.add_argument('--resume', type=str, default=None); parser.add_argument('--no_plot', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ptu.init_gpu(True, gpu_id=args.gpu); device = ptu.device
    print(f"Device: {device}")

    # T5
    print("Loading T5...")
    from transformers import T5Tokenizer, T5EncoderModel
    tok = T5Tokenizer.from_pretrained('t5-large', resume_download=True)
    t5 = T5EncoderModel.from_pretrained('t5-large', resume_download=True).to(device).eval()
    for p in t5.parameters(): p.requires_grad = False

    # Codebook embeddings
    print("Loading MoConVQ...")
    from Script.moconvq_builder import build_agent
    oa = sys.argv[:]; sys.argv = [sys.argv[0], '--experiment_name', 'ft_bandai']
    agent, env = build_agent(gpu=args.gpu); sys.argv = oa
    agent.simple_load('moconvq_base.data', strict=True); agent.eval()

    embed_torch = [torch.cat([bn.embedding, torch.zeros_like(bn.embedding[:2])], dim=0)
                   for bn in agent.posterior.bottle_neck_list]
    embed_torch = [e.to(device) for e in embed_torch]

    # Model
    print("Building model...")
    model = Text2Motion_Transformer(**vars(GPTConfig()), embeddings=embed_torch).to(device)
    ckpt = torch.load('text_generation_GPT.pth', map_location=device)
    sd = {k.replace('module.',''):v for k,v in ckpt.items()} if any(k.startswith('module.') for k in ckpt) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    for i in range(4): model.trans_base.tok_emb[i].weight.requires_grad = False
    # FREEZE trans_temporal — preserves motion quality, only text→token mapping changes
    for param in model.trans_temporal.parameters():
        param.requires_grad = False
    print("Frozen: trans_temporal (motion) + tok_emb (codebook)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,}/{total:,} ({100*trainable/total:.1f}%)")

    # Dataset
    print("\nBuilding datasets...")
    train_ds = BandaiTextMotionDataset(args.dataset_json, args.tokens_npz, 'train',
                                        args.max_seq_len, args.max_depth, tok, t5)
    val_ds = BandaiTextMotionDataset(args.dataset_json, args.tokens_npz, 'val',
                                      args.max_seq_len, args.max_depth, tok, t5)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.1)

    start_epoch = 0; best_val_loss = float('inf'); history = []
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_epoch = ck['epoch']+1; best_val_loss = ck.get('best_val_loss', float('inf'))
        history = ck.get('history', [])

    print(f"\n{'='*60}")
    print(f"Fine-tuning: {args.epochs} epochs, lr={args.lr}, batch={args.batch_size}")
    print(f"Train windows: {len(train_ds)}, Val windows: {len(val_ds)}")
    print(f"{'='*60}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")
        t0 = time.time()
        train_loss, train_ppl = train_epoch(model, train_loader, optimizer, device, epoch+1)
        val_loss, val_ppl = validate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"  Train Loss: {train_loss:.4f} PPL: {train_ppl:.1f} | Val Loss: {val_loss:.4f} PPL: {val_ppl:.1f} | {elapsed:.0f}s")

        history.append({'epoch':epoch, 'train_loss':train_loss, 'train_ppl':train_ppl,
                         'val_loss':val_loss, 'val_ppl':val_ppl})

        ckpt_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch{epoch+1}.pth')
        torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict(),
                     'best_val_loss':best_val_loss, 'history':history}, ckpt_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'epoch':epoch, 'model_state_dict':model.state_dict(),
                         'val_loss':val_loss, 'history':history},
                       os.path.join(args.checkpoint_dir, 'best_model.pth'))
            print(f"  ✓ Best model (val_loss={val_loss:.4f})")

    # Final save
    torch.save({'model_state_dict':model.state_dict(), 'history':history},
               os.path.join(args.checkpoint_dir, 'final_model.pth'))
    with open(os.path.join(args.checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Plot loss curve
    if not args.no_plot:
        plot_loss(history, os.path.join(args.checkpoint_dir, 'loss_curve.png'))

    print(f"\nDone! Checkpoints: {args.checkpoint_dir}")


if __name__ == '__main__':
    main()
