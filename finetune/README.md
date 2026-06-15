# MoConVQ Fine-Tuning: Language-Guided Motion Generation via Zero-Shot Data Construction

## Motivation

MoConVQ (SIGGRAPH 2024) is a state-of-the-art physics-based text-to-motion model. While its pretrained Text2Motion Transformer excels at generating physically plausible motions from simple text prompts (e.g., "a person walks forward"), it struggles with:

- **Abstract/Metaphorical descriptions**: "Walk like a zombie", "Move as if through water"
- **Stylized instructions**: "Sneak forward like a ninja", "Stomp like a sumo wrestler"
- **Negation constraints**: "Walk without swinging your arms"
- **Temporal compositions**: "First walk, then sprint"
- **Zero-action-verb metaphors**: "Conduct a symphony in the empty air with wild baton flourishes"

These are precisely the types of prompts needed for creative applications (games, animation, VR). Our goal: **teach MoConVQ to understand these challenging texts without degrading its motion quality.**

## Approach: Canonical→Difficult Text Mapping

### Core Insight

MoConVQ's pretrained model has **strong zero-shot motion generation** ability — it can generate high-quality motions for clear, physical descriptions. The bottleneck is **text understanding**: it cannot map abstract metaphors to concrete motion parameters.

Our solution: **construct a dataset where each "difficult" text (model CANNOT understand) is paired with a "canonical" text (model CAN understand), and both describe the SAME physical motion.**

```
Difficult text: "Walk like a zombie, arms reaching forward hungrily."
       ↕ (same motion)
Canonical text: "A person walks forward dragging the right leg behind,
                 both arms reaching straight forward, the body swaying
                 heavily side to side."
       ↓
MoConVQ pretrained model → Generates correct zombie-like motion
       ↓
Training pair: (difficult_text, zombie_motion_tokens)
```

### Data Construction Pipeline

We built **279 canonical→difficult text pairs** across 47 action types:

1. **Difficult texts first**: We conceived abstract, metaphorical, negated, and stylized descriptions that challenge the model
2. **Canonical translation**: For each difficult text, we crafted a precise physical description that MoConVQ CAN generate correctly
   - Canonicals follow strict rules: NO negation, NO metaphors, NO abstract language
   - Only concrete physical descriptions ("walks forward with stiff straight legs")
3. **LLM augmentation (DeepSeek)**: Each canonical was expanded to 3-5 difficult variants using 10 linguistic strategies (negation, metaphor, ultra-minimal, extreme detail, etc.)
4. **Motion generation**: All canonical texts → pretrained MoConVQ → physically valid motions → tokenized via VQ-VAE
5. **Normal data**: 2,200 additional simple motion descriptions to prevent catastrophic forgetting

**Final dataset**: 3,032 training pairs (279 canonical × 3 difficult variants + 2,200 normal texts)

### Key Design Rules for Canonical Texts

| Rule | Wrong | Right |
|------|-------|-------|
| No negation | "Walk without swinging arms" | "Walk with arms held straight at sides" |
| No metaphor | "Float like a ghost" | "Walk with very light silent steps" |
| No abstract | "Move gracefully" | "Walk with smooth flowing steps, hips swaying" |
| Specific & physical | "Walk strangely" | "Walk with both feet turned inward, taking short steps" |

## Model Architecture & Fine-Tuning Strategy

### MoConVQ Architecture

```
Text → T5 Encoder → bert_feature (1024-dim)
                           ↓
         ┌─────────────────────────────┐
         │         GPT (Text2Motion)    │
         │                             │
         │  ❄️ trans_temporal (12层)   │ ← FROZEN: motion generation engine
         │     Processes motion latents  │   "how to move" — preserved from pretrained
         │     with cross-attn to text   │
         │                             │
         │  🔥 trans_base (4层)         │ ← TRAINED: text→token mapping
         │     Processes token indices   │   "what token given this text"
         │     with cross-attn to motion │
         │     + tok_emb (frozen)       │
         │                             │
         │  🔥 trans_head (1层)         │ ← TRAINED: prediction head
         │     Logits over codebook      │
         └─────────────────────────────┘
                           ↓
                    Predicted tokens
                           ↓
               VQ-VAE Decoder → Physics Sim → Motion
```

### Why Freeze `trans_temporal`?

Our experiments showed that full-parameter fine-tuning caused the model to develop "hyperactivity" — characters bouncing, twitching, and producing unnatural motions. This is because fine-tuning the motion generation module (`trans_temporal`) disrupted the carefully learned physics priors.

**Solution**: Freeze `trans_temporal` (motion generation, ~150M params), only train `trans_base` + `trans_head` (text→token mapping, ~30M params). The model retains its ability to generate high-quality motions but learns to route new text descriptions to the correct existing motions.

## Experimental Setup

| Parameter | Value |
|-----------|-------|
| Model | Text2Motion_Transformer (193.6M params) |
| Trainable params | ~30M (15.5% of total) |
| Frozen params | ~163M (trans_temporal + tok_emb) |
| Learning rate | 5e-6 (cosine annealing to 5e-7) |
| Batch size | 8 |
| Epochs | 50 |
| Optimizer | AdamW (weight decay=0.01) |
| Max sequence length | 20 frames |
| RVQ depth | 4 layers |

## Results

### Quantitative

| Metric | Pretrained | Fine-tuned (Ours) |
|--------|-----------|-------------------|
| Val Cross-Entropy Loss | 6.24 (random baseline: ln 513) | **5.32** (14.8% improvement) |
| FID (vs canonical motions) | 15.02 | 15.50 |

*Note: FID measures distribution similarity, not text-motion alignment. The fine-tuned model generates more diverse motions matching difficult texts, which slightly increases FID while dramatically improving qualitative alignment.*

### Qualitative — Demo Samples

| Text | Pretrained | Our Model |
|------|-----------|-----------|
| "Walk as if pushing against a fierce hurricane wind..." | [t9_wind_pretrained.gif](demo/t9_wind_pretrained.gif) | [t9_wind_frozen.gif](demo/t9_wind_frozen.gif) |
| "Creep ahead with feline grace, body low and coiled..." | [za3_cat_stalk_pretrained.gif](demo/za3_cat_stalk_pretrained.gif) | [za3_cat_stalk_frozen.gif](demo/za3_cat_stalk_frozen.gif) |
| "Conduct a symphony in the empty air with wild baton flourishes." | [za5_conductor_pretrained.gif](demo/za5_conductor_pretrained.gif) | [za5_conductor_frozen.gif](demo/za5_conductor_frozen.gif) |

### Key Findings

1. **Zero-action-verb metaphors benefit most**: Texts with no concrete action words (e.g., "charge like a T-Rex" instead of "run heavily") show the largest improvement
2. **Motion quality preserved**: Freezing `trans_temporal` eliminated the "hyperactivity" problem seen in full fine-tuning
3. **Modest but meaningful improvement**: 14.8% val loss reduction, with clear qualitative gains on abstract/stylized prompts

## Reproducing Demo Samples

### Prerequisites

#### 1. MoConVQ Environment
Follow the main project README to set up the conda environment, compile VclSimuBackend, and install dependencies.

#### 2. Pretrained Weights (download from official sources)

| File | Size | Description | Download Link |
|------|------|-------------|---------------|
| `moconvq_base.data` | 242 MB | VQ-VAE motion encoder/decoder | [PKU Disk](https://disk.pku.edu.cn/link/AAAFE3B2DDB1AC420EB5C4E0910196116F) or [OneDrive](https://1drv.ms/f/s!AsrkHbtkj4LsbqMZI08Bt9jFPJ4?e=SXkFlg) |
| `text_generation_GPT.pth` | 740 MB | Pretrained Text2Motion Transformer | Same as above |
| `T5-large` | ~3 GB | Text encoder (auto-downloaded) | [HuggingFace](https://huggingface.co/google-t5/t5-large) (HF mirror: set `HF_ENDPOINT=https://hf-mirror.com`) |

Place `moconvq_base.data` and `text_generation_GPT.pth` in the project root directory (alongside this `finetune/` folder).

#### 3. Our Fine-tuned Model
- Download `best_model.pth` from [Google Drive / PKU Disk link]
- Place it in `model/best_model.pth`

### Generate demo motions
```bash
cd MoConVQ-main
python scripts/demo_generate.py
```
Outputs BVH files in `demo_out/`.

### Render GIFs
```bash
python scripts/bvh_to_gif_fast.py demo_out/t9_wind_frozen.bvh
```
Or use the official renderer:
```bash
python -c "
from Visualization.bvh_visualizer import visualize_bvh
visualize_bvh('demo_out/t9_wind_frozen.bvh', 'output.gif', output_fps=20)
"
```

## File Structure

```
finetune/
├── README.md
├── model/best_model.pth           # Fine-tuned model (frozen trans_temporal)
├── data/
│   ├── canonical_pairs.json       # 279 canonical→difficult pairs
│   └── normal_texts.json          # 2,200 normal texts
├── demo/
│   ├── texts.json                 # 3 demo texts
│   └── *_pretrained.{bvh,gif}     # Pretrained outputs
│   └── *_frozen.{bvh,gif}         # Fine-tuned outputs
├── results/
│   ├── loss_curve.png             # Training curve
│   └── fid_results.json           # FID evaluation
└── scripts/
    ├── finetune_bandai.py          # Fine-tuning script
    ├── batch_generate_fast.py      # Batch motion generation
    ├── bvh_remapper.py             # BVH skeleton remapping
    ├── bvh_to_gif_fast.py          # Fast GIF renderer
    ├── deepseek_augment_skill.py   # LLM text augmentation
    └── demo_generate.py            # Demo reproduction script
```

## Summary

We present a method to fine-tune MoConVQ's Text2Motion Transformer using a novel **canonical→difficult text mapping approach**. By freezing the motion generation module (`trans_temporal`) and only training the text→token mapping layers (`trans_base` + `trans_head`), we achieve improved understanding of abstract, metaphorical, and stylized text prompts while preserving motion quality. Our data construction pipeline leverages the pretrained model's zero-shot capability combined with LLM-based text augmentation to generate 3,032 high-quality training pairs covering 47 action types.
