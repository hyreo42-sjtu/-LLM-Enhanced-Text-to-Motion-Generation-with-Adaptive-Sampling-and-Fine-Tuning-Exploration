# LLM-Enhanced Text-to-Motion Generation with Adaptive Sampling and Fine-Tuning

Two-person project for language-guided 3D character motion generation, built on [MoConVQ](https://github.com/yyyokoo/MoConVQ) (SIGGRAPH 2024).

## Repository Structure

```
├── LLM_Enhance/                       # Ding Hao: LLM pipeline + sampling + visualization
│   ├── Generation/                # Inference pipeline
│   │   ├── generate.py            # Main entry: text → BVH + GIF + 3D view
│   │   ├── prompt_engine.py       # DeepSeek-V4 API + strategy mapping + fallback
│   │   ├── llm_prompt.py          # System prompt (motion semantics)
│   │   ├── llm_config_example.py  # API config template → copy to llm_config.py
│   │   └── sampling_strategies.py # 5 preset T/k/p configurations
│   └── Visualization/             # Rendering tools
│       ├── ode_viewer.py          # ODE drawstuff OpenGL 3D viewer
│       ├── bvh_visualizer.py      # Single-angle stick-figure GIF
│       └── multi_angle_viewer.py  # 4-angle 2×2 GIF
│
└── finetune/                      # Huang Youran: fine-tuning exploration
    ├── scripts/                   # Training + data processing scripts
    ├── data/                      # Bandai Namco text-motion pairs
    ├── demo/                      # Demo BVH + GIF outputs
    └── results/                   # FID metrics, loss curves
```

## Setup

1. Clone MoConVQ into the parent directory
2. `cp LLM_Enhance/Generation/llm_config_example.py LLM_Enhance/Generation/llm_config.py` and add your API key
3. `conda install -c conda-forge freeglut glew` (for OpenGL 3D viewer)

## Usage

```bash
# LLM pipeline (Ding Hao)
python LLM_Enhance/Generation/generate.py --prompt "A person kicks."          # generate
python LLM_Enhance/Generation/generate.py --prompt "A person dances." --viewer # + 3D view
python LLM_Enhance/Generation/generate.py --prompt "A person kicks." --record  # + 3D GIF

# Fine-tuning (Huang Youran)
# See finetune/README.md
```

## Key Results

- **92%** zero-shot strategy classification accuracy (DeepSeek-V4)
- 5 adaptive sampling strategies via LLM semantic analysis
- Fine-tuning explored on augmented Bandai Namco data (overfitting identified)
- Native ODE OpenGL 3D viewer with volumetric rendering
