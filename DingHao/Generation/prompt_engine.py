"""LLM-based prompt expansion and sampling strategy selection.

Uses DeepSeek API to:
1. Expand short user prompts into detailed motion descriptions
2. Extract semantic features (4 yes/no questions) for strategy selection

Falls back to rule-based keyword matching if the API is unavailable.
"""

import json
import re
import warnings
from typing import Dict, Optional

import requests

try:
    from DingHao.Generation.llm_config import (
        DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
        LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
    )
    from DingHao.Generation.llm_prompt import SYSTEM_PROMPT, FEATURE_KEYS, LEGACY_NUMERIC_KEYS
except ModuleNotFoundError:
    try:
        # Running from DingHao/Generation/
        from DingHao.Generation.llm_config import (
            DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
            LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
        )
        from DingHao.Generation.llm_prompt import SYSTEM_PROMPT, FEATURE_KEYS, LEGACY_NUMERIC_KEYS
    except ModuleNotFoundError:
        from DingHao.Generation.llm_config_example import (
            DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DEEPSEEK_MODEL,
            LLM_TEMPERATURE, LLM_MAX_TOKENS, LLM_TIMEOUT,
        )
        from DingHao.Generation.llm_prompt import SYSTEM_PROMPT, FEATURE_KEYS, LEGACY_NUMERIC_KEYS
        import warnings
        warnings.warn(
            'Using example API key. Copy Generation/llm_config.example.py '
            'to Generation/llm_config.py with your real DeepSeek API key.'
        )
    from llm_prompt import SYSTEM_PROMPT, FEATURE_KEYS, LEGACY_NUMERIC_KEYS


# ── Semantic features → strategy mapping ─────────────────────────────

def _answers_to_strategy(specific: bool, abstract: bool, flow: bool, energetic: bool) -> str:
    """Deterministic mapping from yes/no motion features to sampling strategy.

    Principles (in priority order):
    1. Specific trajectory → deterministic.
       Physical actions (kick, punch, throw, point) have essentially one
       correct trajectory. Low T + small k to find the "right answer".
    2. Abstract/open-ended → creative.
       No single correct motion exists. High T + nucleus sampling for
       diverse, surprising output.
    3. Continuous flow + NOT energetic → smooth.
       Frame-to-frame consistency matters most. Low T for smooth transitions.
    4. Energetic + NOT specific → expressive.
       Dynamic motions benefit from adaptive top_p sampling.
    5. Otherwise → moderate (safe default, general locomotion).
    """
    if specific:
        return "deterministic"
    if abstract:
        return "creative"
    if flow and not energetic:
        return "smooth"
    if energetic:
        return "expressive"
    return "moderate"


# ── Rule-based fallback ──────────────────────────────────────────────

# (keyword_list, strategy_name)
# NOTE: more specific rules (smooth) come first to avoid ambiguous single-char
# Chinese matches (e.g. "打" in "打太极拳" = practice, not hit)
_KEYWORD_RULES = [
    (["ballet", "tai chi", "taiji", "太极拳", "太极", "swim", "swims", "swimming",
      "glide", "glides", "gliding", "sneak", "sneaks", "sneaking",
      "tiptoe", "tiptoes", "tiptoeing", "float", "floats", "floating",
      "yoga", "芭蕾", "游泳", "瑜伽",
      "忍者", "悄悄", "踮脚", "漂浮", "freestyle", "ninja"], "smooth"),
    (["kick", "kicks", "kicking", "punch", "punches", "punching",
      "grab", "grabs", "grabbing", "point", "points", "pointing",
      "throw", "throws", "throwing", "push", "pushes", "pushing",
      "pull", "pulls", "pulling", "press", "presses", "pressing",
      "lift", "lifts", "lifting", "pick up", "wave", "waves", "waving",
      "踢", "拳击", "抓", "扔", "推", "拉", "举", "挥手", "出拳"], "deterministic"),
    (["dance", "dances", "dancing", "jump", "jumps", "jumping",
      "cheer", "cheers", "cheering", "spin", "spins", "spinning",
      "run", "runs", "running", "sprint", "sprints", "sprinting",
      "leap", "leaps", "leaping", "hop", "hops", "hopping",
      "skip", "skips", "skipping",
      "跳舞", "跳跃", "跑", "冲刺", "旋转", "欢呼"], "expressive"),
    (["express", "expresses", "expressing", "emotion", "emotions",
      "feeling", "feelings", "surprise", "surprises", "surprised",
      "abstract", "creative", "imagine", "imagines", "imagining",
      "fantasy", "高兴", "表达", "情感", "创意", "想象", "惊喜", "害怕", "生气", "难过"], "creative"),
]
_DEFAULT_STRATEGY = "moderate"


def _rule_based_classify(prompt: str) -> Dict:
    """Simple keyword-based classification fallback.

    English keywords use word-boundary matching with common inflection
    suffixes (s/es/ing/ed) to avoid substring false positives.
    Chinese keywords use direct substring matching.
    """
    lower = prompt.lower()
    for keywords, strategy in _KEYWORD_RULES:
        for kw in keywords:
            if kw.isascii():
                if re.search(r'\b' + re.escape(kw) + r'(?:es|s|ing|ed)?\b', lower):
                    return {
                        "expanded_prompt": prompt,
                        "strategy": strategy,
                        "reason": f"Rule-based fallback: keyword '{kw}' matched → {strategy}",
                    }
            else:
                if kw in lower:
                    return {
                        "expanded_prompt": prompt,
                        "strategy": strategy,
                        "reason": f"Rule-based fallback: keyword '{kw}' matched → {strategy}",
                    }
    return {
        "expanded_prompt": prompt,
        "strategy": _DEFAULT_STRATEGY,
        "reason": "Rule-based fallback: no keyword matched → moderate (default)",
    }


# ── JSON extraction ──────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[Dict]:
    """Robust JSON extraction from LLM output (handles markdown fences etc.)."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to extract from ```json ... ``` block
    fence_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find the first { ... } pair
    brace_match = re.search(r'\{[\s\S]*\}', text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── DeepSeek API call ────────────────────────────────────────────────

def _call_deepseek(prompt: str) -> str:
    """Send prompt to DeepSeek API, return response text.

    Raises requests.RequestException on failure.
    """
    response = requests.post(
        DEEPSEEK_API_URL,
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": LLM_TEMPERATURE,
            "max_tokens": LLM_MAX_TOKENS,
        },
        timeout=LLM_TIMEOUT,
        proxies={"http": None, "https": None},  # bypass Windows system proxy
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


# ── PromptEngine ─────────────────────────────────────────────────────

class PromptEngine:
    """LLM-driven prompt expansion and strategy selection via DeepSeek API."""

    def process(self, prompt: str) -> Dict:
        """Expand prompt and select sampling strategy.

        Returns dict with keys: expanded_prompt, strategy, reason
        """
        # Try DeepSeek API first
        try:
            raw = _call_deepseek(prompt)
        except Exception as e:
            warnings.warn(
                f"[PromptEngine] DeepSeek API call failed: {e}\n"
                f"  Falling back to rule-based keyword matching."
            )
            return _rule_based_classify(prompt)

        result = _extract_json(raw)
        if result is None:
            warnings.warn(
                f"[PromptEngine] Failed to parse API response as JSON:\n{raw}\n"
                f"Falling back to rule-based classification."
            )
            return _rule_based_classify(prompt)

        result.setdefault("reason", "")

        if "expanded_prompt" not in result:
            return _rule_based_classify(prompt)

        # New format: yes/no questions → deterministic mapping (primary path)
        if all(k in result for k in FEATURE_KEYS):
            try:
                s = bool(result["specific_trajectory"])
                a = bool(result["abstract_open"])
                f = bool(result["continuous_flow"])
                e = bool(result["energetic_dynamic"])
                result["strategy"] = _answers_to_strategy(s, a, f, e)
                if not result.get("reason"):
                    result["reason"] = (
                        f"specific={s} abstract={a} flow={f} energetic={e} → {result['strategy']}"
                    )
            except (ValueError, TypeError):
                return _rule_based_classify(prompt)

        # Legacy numeric format (backward compatible)
        elif all(k in result for k in LEGACY_NUMERIC_KEYS):
            try:
                p = int(result["precision"])
                e = int(result["energy"])
                f = int(result["fluidity"])
                p, e, f = max(1, min(5, p)), max(1, min(5, e)), max(1, min(5, f))
                result["strategy"] = _answers_to_strategy(
                    specific=(p >= 4),
                    abstract=(p <= 2),
                    flow=(f >= 4 and e <= 3),
                    energetic=(e >= 4),
                )
            except (ValueError, TypeError):
                return _rule_based_classify(prompt)

        # Legacy direct strategy name (backward compatible)
        elif "strategy" in result:
            valid = {"deterministic", "moderate", "smooth", "expressive", "creative"}
            if result["strategy"] not in valid:
                warnings.warn(
                    f"[PromptEngine] Invalid strategy '{result['strategy']}', "
                    f"falling back to rule-based."
                )
                return _rule_based_classify(prompt)

        else:
            return _rule_based_classify(prompt)

        return result


# ── Standalone test ──────────────────────────────────────────────────

if __name__ == "__main__":
    test_prompts = [
        "A person kicks with their right leg.",
        "A person is jogging forward.",
        "A person dances energetically.",
        "A person does a ballet spin.",
        "A person expresses joy through movement.",
        "一个人打太极拳。",
    ]
    engine = PromptEngine()
    for p in test_prompts:
        result = engine.process(p)
        print(f"\nInput:    {p}")
        print(f"Expanded: {result['expanded_prompt']}")
        print(f"Strategy: {result['strategy']}")
        print(f"Reason:   {result.get('reason', 'N/A')}")
