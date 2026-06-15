#!/usr/bin/env python3
"""
DeepSeek-powered text augmentation skill for MoConVQ fine-tuning data construction.

This skill generates DIVERSE text variants for a given canonical motion description.
Each canonical text describes a motion that MoConVQ CAN generate correctly.
The variants are "difficult" texts — the same motion described in ways MoConVQ CANNOT
understand (negation, abstraction, complex composition, etc.).

Architecture:
  Canonical text (model CAN generate) → MoConVQ → Motion (ground truth)
  + 10 difficult variants (model CAN'T generate) → paired with same motion
  → Fine-tune to teach model the mapping

Usage:
  python Script/deepseek_augment_skill.py --canonical "Walk stiffly" --gap "negation"
  python Script/deepseek_augment_skill.py --batch canonical_texts.json --output variants.json
"""

import sys, os, json, argparse, time, requests

# ══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPT — the core of the skill
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_TEMPLATE = """You are an expert in text-to-motion data augmentation. Your task is to generate DIVERSE text descriptions of THE SAME PHYSICAL MOTION, varying the linguistic style to challenge a motion generation AI.

## THE MOTION
Canonical description: "{canonical}"
Gap direction: {gap_direction}
Gap explanation: {gap_explanation}

## WHAT YOU MUST PRESERVE
The PHYSICAL MOTION described by all variants MUST be identical to the canonical text:
- Same action (walking/running/standing/waving/kicking/etc.)
- Same speed, rhythm, and intensity
- Same body parts involved
- Same spatial trajectory

## WHAT YOU MUST VARY
Generate 10 variants using the following strategies (use each strategy exactly once):

### Strategy 1 — Negation (否定表达)
Describe the motion by saying what NOT to do. Use words like "without", "do not", "avoid", "never".
Example: "Walk forward" → "Move forward without letting your arms swing at all."

### Strategy 2 — Abstract Metaphor (抽象隐喻)
Use poetic, metaphorical, or figurative language. Compare the motion to something in nature, art, or daily life.
Example: "Walk carefully" → "Move as if the floor beneath you is made of eggshells."

### Strategy 3 — High-Precision Detail (高精度描述)
Add specific biomechanical details: joint angles, muscle groups, exact timing, precise distances.
Example: "Walk slowly" → "Step forward at 0.5 meters per second, keeping the knee flexion angle between 10 and 15 degrees during stance phase."

### Strategy 4 — Casual / Conversational (口语化)
Describe the motion in everyday spoken language, as if telling a friend what to do.
Example: "Walk proudly" → "Just strut around like you own the place, you know?"

### Strategy 5 — Instructional / Imperative (指令式)
Use direct commands. Address the reader as "you". Use short sentences.
Example: "Raise left hand" → "You. Left hand. Lift it up. Now wave it slowly. Right hand stays down."

### Strategy 6 — Emotional Narrative (情感叙事)
Embed the motion in an emotional context. Describe the feeling behind the movement.
Example: "Walk tiredly" → "After the longest day of your life, you drag yourself home, every step a battle against exhaustion."

### Strategy 7 — Ultra-Minimal (极简)
Describe the motion in 3-5 words. Keep it terse but unambiguous.
Example: "Walk without arm swing" → "Walk. Arms still. No swinging."

### Strategy 8 — Extreme Detail (极度详细)
Use 25+ words. Add environmental context, sensory details, exact timing, body awareness. Be cinematic.
Example: "Walk smoothly" → "Glide across the polished marble floor with a fluid heel-to-toe transition, your hips remaining perfectly level as if balancing an invisible book on your head, each footfall silent and deliberate."

### Strategy 9 — Question / Hypothetical (疑问式)
Frame the motion as a question, a challenge, or a "what if" scenario.
Example: "Walk slowly" → "Can you move forward at the pace of a hesitant snail, each step a deliberate negotiation with gravity?"

### Strategy 10 — Constraint Stacking (约束叠加)
Combine multiple constraints: speed + body part + direction + quality simultaneously.
Example: "Kick with right leg" → "While balancing perfectly on your left foot, snap your right leg forward from the hip, toes pointed, returning to stance without stumbling."

## OUTPUT FORMAT
Output EXACTLY 10 lines, each line starting with the strategy number and a period:
1. [Strategy name] variant text here...
2. [Strategy name] variant text here...
...
10. [Strategy name] variant text here...

No extra text, no explanations, no markdown formatting. Just the 10 numbered lines."""


# ══════════════════════════════════════════════════════════════════════════════
# GAP DIRECTION DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

GAP_DEFINITIONS = {
    "时序组合": {
        "explanation": "The model cannot understand sequential instructions like 'first A then B'. It only executes the last part or merges them into one generic action.",
        "strategies": ["negation", "abstract_metaphor", "detailed", "casual", "instructional", "emotional", "minimal", "extreme_detail", "question", "constraint_stacking"],
    },
    "否定约束": {
        "explanation": "The model ignores negative constraints like 'do not swing arms' or 'without moving your head'. It defaults to the natural version of the action with all typical secondary motions.",
        "strategies": ["negation", "abstract_metaphor", "detailed", "casual", "instructional", "emotional", "minimal", "extreme_detail", "question", "constraint_stacking"],
    },
    "异步约束": {
        "explanation": "The model cannot understand asymmetric instructions where only one side of the body moves. It tends to move both sides symmetrically.",
        "strategies": ["negation", "abstract_metaphor", "detailed", "casual", "instructional", "emotional", "minimal", "extreme_detail", "question", "constraint_stacking"],
    },
    "抽象隐喻": {
        "explanation": "The model cannot map figurative or metaphorical descriptions to concrete physical motions. 'Walk like on thin ice' gets misinterpreted as crouching or freezing.",
        "strategies": ["negation", "abstract_metaphor", "detailed", "casual", "instructional", "emotional", "minimal", "extreme_detail", "question", "constraint_stacking"],
    },
    "物体交互": {
        "explanation": "The model cannot depict interaction with imaginary objects (doors, boxes, tools). The motion becomes a generic hand wave or unrelated action.",
        "strategies": ["negation", "abstract_metaphor", "detailed", "casual", "instructional", "emotional", "minimal", "extreme_detail", "question", "constraint_stacking"],
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def generate_variants(canonical_text, gap_direction, api_key, model="deepseek-chat"):
    """Generate 10 diverse text variants for a canonical motion description.

    Args:
        canonical_text: The canonical description (model CAN generate this correctly)
        gap_direction: One of the gap categories (时序组合/否定约束/etc.)
        api_key: DeepSeek API key

    Returns:
        list of 10 variant text strings, or None on failure
    """
    gap_info = GAP_DEFINITIONS.get(gap_direction, GAP_DEFINITIONS["抽象隐喻"])

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        canonical=canonical_text,
        gap_direction=gap_direction,
        gap_explanation=gap_info["explanation"],
    )

    user_prompt = f"Generate 10 diverse text variants for: \"{canonical_text}\" ({gap_direction})"

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(url, headers=headers, json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.95,
            "max_tokens": 1200,
        }, timeout=60)

        if resp.status_code != 200:
            print(f"  API error {resp.status_code}: {resp.text[:100]}")
            return None

        content = resp.json()['choices'][0]['message']['content']
        variants = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line: continue
            # Parse "N. [strategy] text..."
            if line[0].isdigit() and '. ' in line:
                text = line.split('. ', 1)[1]
                # Remove strategy label in brackets if present
                if text.startswith('[') and '] ' in text:
                    text = text.split('] ', 1)[1]
                text = text.strip().strip('"').strip("'")
                if len(text) > 5:
                    variants.append(text)

        return variants[:10] if len(variants) >= 5 else None

    except Exception as e:
        print(f"  Error: {e}")
        return None


def augment_batch(canonical_texts_dict, api_key, output_path, gap_col="gap", text_col="canonical"):
    """Batch augment multiple canonical texts.

    Args:
        canonical_texts_dict: {key: {gap_col: "...", text_col: "..."}}
        api_key: DeepSeek API key
        output_path: where to save the JSON output
        gap_col: key for gap direction in the input dict
        text_col: key for canonical text in the input dict

    Returns:
        dict: {key: {"canonical": "...", "gap": "...", "variants": [...]}}
    """
    results = {}
    total = len(canonical_texts_dict)

    for i, (key, info) in enumerate(canonical_texts_dict.items()):
        canonical = info.get(text_col, info) if isinstance(info, dict) else info
        gap = info.get(gap_col, "抽象隐喻") if isinstance(info, dict) else "抽象隐喻"

        print(f"\n[{i+1}/{total}] {key}")
        print(f"  Canonical: \"{canonical[:80]}...\"")
        print(f"  Gap: {gap}")

        variants = generate_variants(canonical, gap, api_key)
        if variants:
            results[key] = {
                "canonical": canonical,
                "gap": gap,
                "variants": variants,
            }
            print(f"  ✓ Generated {len(variants)} variants")
            for j, v in enumerate(variants):
                print(f"    {j+1}. \"{v[:100]}{'...' if len(v)>100 else ''}\"")
        else:
            print(f"  ✗ Failed")
            results[key] = {
                "canonical": canonical,
                "gap": gap,
                "variants": [canonical],  # fallback to canonical only
            }

        time.sleep(0.3)

    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(results)} entries to {output_path}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='DeepSeek text augmentation for MoConVQ fine-tuning')
    parser.add_argument('--canonical', type=str, help='Single canonical text to augment')
    parser.add_argument('--gap', type=str, default='抽象隐喻',
                       choices=list(GAP_DEFINITIONS.keys()),
                       help='Gap direction category')
    parser.add_argument('--batch', type=str, help='JSON file with batch of canonical texts')
    parser.add_argument('--output', type=str, default='out/finetune_v2/variants.json',
                       help='Output JSON file')
    parser.add_argument('--api-key', type=str, required=True, help='DeepSeek API key')
    args = parser.parse_args()

    if args.canonical:
        # Single mode
        variants = generate_variants(args.canonical, args.gap, args.api_key)
        if variants:
            print(f"\nGenerated {len(variants)} variants:")
            for i, v in enumerate(variants):
                print(f"  {i+1}. {v}")
        else:
            print("Failed to generate variants")

    elif args.batch:
        # Batch mode
        with open(args.batch) as f:
            data = json.load(f)
        augment_batch(data, args.api_key, args.output)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
