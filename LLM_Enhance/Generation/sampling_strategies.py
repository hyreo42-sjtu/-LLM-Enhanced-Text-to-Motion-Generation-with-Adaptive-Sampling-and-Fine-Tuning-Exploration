"""Preset sampling strategies for MoConGPT motion generation.

Each strategy encodes a principled combination of temperature, top_k, and top_p
tailored to a specific class of motion prompts. The LLM prompt engine selects
among these presets based on semantic analysis of the input prompt.

Principles:
  - temperature sharpens (T<1) or flattens (T>1) the logit distribution
  - top_k provides a fixed-size hard truncation filter
  - top_p provides an adaptive truncation that respects distribution shape
  - top_k and top_p SHOULD NOT be combined; each strategy uses one or the other
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0                # 0 = disabled, use top_p or raw sampling
    top_p: Optional[float] = None # None = disabled, use top_k or raw sampling


PRESETS: Dict[str, SamplingConfig] = {
    # Single clear action (kick, punch, point).
    # Very low T sharpens the distribution; small k locks in high-confidence codes.
    "deterministic": SamplingConfig(temperature=0.4, top_k=15),

    # General locomotion (walk, jog, sit). Safe default.
    # T=0.6 slightly reduced from 0.7 — DeepSeek's accurate classification means
    # non-moderate motions won't leak into this strategy, so we can tighten it.
    "moderate":      SamplingConfig(temperature=0.6, top_k=50),

    # Continuous graceful motion (ballet, tai chi, swim).
    # Low T ensures frame-to-frame code consistency; k=30 is tighter than default.
    "smooth":        SamplingConfig(temperature=0.5, top_k=30),

    # Energetic large-amplitude actions (dance, jump, cheer).
    # T=1 keeps the native distribution; top_p=0.9 adaptively filters the long tail.
    "expressive":    SamplingConfig(temperature=1.0, top_p=0.90),

    # Abstract / emotional / open-ended prompts (express joy, move creatively).
    # T>1 deliberately flattens the distribution to explore low-probability codes;
    # large p=0.95 keeps more candidates. Physics simulation provides a safety net.
    "creative":      SamplingConfig(temperature=1.3, top_p=0.95),
}

DEFAULT_STRATEGY = "moderate"


def get_strategy(name: str) -> SamplingConfig:
    """Resolve a strategy name to its SamplingConfig, falling back to moderate."""
    return PRESETS.get(name, PRESETS[DEFAULT_STRATEGY])


if __name__ == "__main__":
    print("Preset sampling strategies:\n")
    for name, cfg in PRESETS.items():
        k_str = str(cfg.top_k) if cfg.top_k else "-"
        p_str = f"{cfg.top_p:.2f}" if cfg.top_p is not None else "-"
        print(f"  {name:>14s}:  T={cfg.temperature:.1f}  "
              f"top_k={k_str:>4s}  top_p={p_str:>4s}")
    print(f"\n  default: {DEFAULT_STRATEGY}")
