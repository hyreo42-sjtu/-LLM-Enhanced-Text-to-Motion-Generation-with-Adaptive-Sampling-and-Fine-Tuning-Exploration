# LLM-Enhanced Adaptive Sampling for Text-to-Motion Generation

Ding Hao's contribution: integrating DeepSeek-V4 semantic analysis into MoConVQ for adaptive sampling strategy selection, plus native ODE OpenGL 3D visualization.

## Structure

```
LLM_Enhance/
├── Generation/                    # Inference pipeline
│   ├── generate.py                # Main entry point
│   ├── prompt_engine.py           # DeepSeek API + strategy mapping + keyword fallback
│   ├── llm_prompt.py              # System prompt (4 yes/no motion questions)
│   ├── llm_config_example.py      # API config template → copy to llm_config.py
│   └── sampling_strategies.py     # 5 preset T/k/p sampling configurations
└── Visualization/                 # Rendering tools
    ├── ode_viewer.py              # ODE drawstuff OpenGL 3D volumetric renderer
    ├── bvh_visualizer.py          # Single-angle matplotlib stick-figure GIF
    └── multi_angle_viewer.py      # 4-angle 2×2 synchronized GIF
```

## Pipeline

```
User Prompt → DeepSeek-V4 (expand + classify) → T5 encode → MoConGPT (adaptive T/k/p) → ODE physics → BVH + GIF + 3D view
```

## Sampling Strategies

| Strategy | T | top_k | top_p | Target Motion |
|----------|---|-------|-------|---------------|
| deterministic | 0.4 | 15 | — | kick, punch, throw |
| moderate | 0.6 | 50 | — | walk, jog (default) |
| smooth | 0.5 | 30 | — | ballet, tai chi, swim |
| expressive | 1.0 | — | 0.90 | dance, jump, cheer |
| creative | 1.3 | — | 0.95 | express joy, act surprised |

## Usage

```bash
# Generate motion
python LLM_Enhance/Generation/generate.py --prompt "A person kicks."

# With OpenGL 3D viewer
python LLM_Enhance/Generation/generate.py --prompt "A person dances." --viewer

# Record 3D GIF
python LLM_Enhance/Generation/generate.py --prompt "A person kicks." --record
```

## Key Results

- **92%** zero-shot strategy classification accuracy (DeepSeek-V4 vs 75% Qwen 1.5B baseline)
- 12 diverse prompts: English + Chinese, concrete + abstract
- 5 strategies actively used across the test suite
- Native ODE OpenGL 3D viewer with volumetric capsule/cylinder rendering
- 1,300 LLM-augmented Bandai Namco pairs for fine-tuning exploration
