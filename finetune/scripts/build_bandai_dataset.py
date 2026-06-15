#!/usr/bin/env python3
"""
Build text-motion dataset from Bandai Namco Research Motion Dataset.

Pipeline:
1. Extract BVH files with style labels from Bandai Namco
2. Use DeepSeek API to generate diverse text descriptions (5-10 per motion)
3. Tokenize BVH files to codebook tokens via MoConVQ
4. Assemble final text-motion pairs for fine-tuning

Usage:
  python Script/build_bandai_dataset.py --api-key YOUR_KEY
  python Script/build_bandai_dataset.py --api-key YOUR_KEY --skip-augment --skip-tokenize
"""

import os, sys, json, re, argparse, time, zipfile
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'ModifyODESrc'))
sys.path.insert(0, os.path.join(BASE_DIR, 'diff-quaternion', 'TorchRotation'))


# ─── CONFIG ───
BANDAI_ZIP = os.path.join(BASE_DIR, 'dataset', 'mocap_data', 'unified_bvh', 'bandai_namco_motion.zip')
OUTPUT_DIR = os.path.join(BASE_DIR, 'out', 'bandai_dataset')
BVH_OUT_DIR = os.path.join(OUTPUT_DIR, 'bvh')
TOKENS_OUT = os.path.join(OUTPUT_DIR, 'tokens.npz')
DATASET_OUT = os.path.join(OUTPUT_DIR, 'text_motion_dataset.json')

# Style-to-template mapping for basic literal conversion
STYLE_ACTION_TEMPLATES = {
    'active': 'a person {action} actively and energetically',
    'angry': 'a person {action} angrily',
    'childish': 'a person {action} in a childish manner',
    'chimpira': 'a person {action} like a street punk',
    'elderly': 'an elderly person {action}ing slowly',
    'exhausted': 'a person {action} exhaustedly, barely able to continue',
    'feminine': 'a person {action} with a feminine gait',
    'giant': 'a person {action} like a giant, heavy and lumbering',
    'happy': 'a person {action} happily',
    'masculine': 'a person {action} with a masculine stance',
    'masculinity': 'a person {action} in a very masculine way',
    'musical': 'a person {action} with a musical rhythm',
    'normal': 'a person {action}ing normally',
    'not-confident': 'a person {action} without confidence, hesitantly',
    'old': 'an old person {action}ing',
    'proud': 'a person {action} proudly, with chest out',
    'sad': 'a person {action} sadly, with a downcast posture',
    'tired': 'a person {action} tiredly',
    'youthful': 'a person {action} like an energetic youth',
}

# Action name normalization
ACTION_MAP = {
    'raise-up-both-hands': 'raising both hands',
    'dance-long': 'dancing for a long time',
    'dance-short': 'dancing briefly',
}


def parse_filename(filename):
    """Parse Bandai Namco filename to extract action and style."""
    basename = os.path.splitext(os.path.basename(filename))[0]
    match = re.match(r'dataset-\d+_(.+?)_(\S+?)_(\d+)', basename)
    if match:
        action_raw = match.group(1)
        style = match.group(2)
        take = int(match.group(3))
        action = ACTION_MAP.get(action_raw, action_raw.replace('-', ' '))
        return action, style, take
    return None, None, None


def generate_literal_text(action, style):
    """Generate the basic literal text from action+style."""
    template = STYLE_ACTION_TEMPLATES.get(style, f'a person {{action}} with {style} style')
    return template.format(action=action)


# ─── STEP 1: CATALOG FILES ───
def catalog_files():
    """Extract file listing from Bandai Namco zip."""
    print("=" * 60)
    print("[1/4] Cataloging Bandai Namco BVH files...")

    files = []
    with zipfile.ZipFile(BANDAI_ZIP, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.bvh') and '__MACOSX' not in name:
                action, style, take = parse_filename(name)
                if action and style:
                    files.append({
                        'zip_path': name,
                        'action': action,
                        'style': style,
                        'take': take,
                        'basename': os.path.basename(name),
                    })

    # Group by (action, style)
    groups = defaultdict(list)
    for f in files:
        groups[(f['action'], f['style'])].append(f)

    print(f"  Total BVH files: {len(files)}")
    print(f"  Unique (action, style) pairs: {len(groups)}")

    # Print summary
    actions = defaultdict(list)
    for (action, style), items in groups.items():
        actions[action].append((style, len(items)))

    for action, styles in sorted(actions.items()):
        style_str = ', '.join(f'{s}({c})' for s, c in sorted(styles))
        print(f"    {action}: {style_str}")

    return files, groups


# ─── STEP 2: TEXT AUGMENTATION ───
def augment_texts_with_deepseek(texts_info, api_key):
    """Use DeepSeek API to generate diverse text descriptions."""
    import requests

    print(f"\n{'=' * 60}")
    print(f"[2/4] Augmenting texts with DeepSeek API...")
    print(f"  Processing {len(texts_info)} unique (action, style) pairs")

    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    results = {}
    processed = 0

    for (action, style), info in texts_info.items():
        literal = info['literal']
        key = f"{action}_{style}"

        system_prompt = (
            "You are a motion description expert. Your task is to generate DIVERSE text descriptions "
            "of the SAME physical human motion for training a text-to-motion AI model.\n\n"
            "RULES:\n"
            "1. ALL descriptions must describe the EXACT same motion action and style\n"
            f"2. The base motion is: action='{action}', style='{style}'\n"
            '3. Vary the wording significantly: different sentence structures, vocabulary, detail level\n'
            '4. Include descriptions of varying lengths (5-30 words)\n'
            '5. Some descriptions should be simple ("a person walking proudly")\n'
            '6. Some should be detailed ("an individual strides forward with their chest puffed out...\")\n'
            '7. Some should use different perspectives (first-person, observer, cinematic)\n'
            '8. Do NOT change the core action or style — only the wording\n'
            '9. Output EXACTLY one description per line, numbered 1 through N\n'
            '10. No extra text before or after the numbered list'
        )

        user_prompt = (
            f"Base description: \"{literal}\"\n\n"
            f"Generate 6 diverse text descriptions of this motion "
            f"(action: {action}, style: {style}). "
            f"Include short, medium, and long descriptions with varied wording."
        )

        try:
            response = requests.post(
                url, headers=headers,
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.9,
                    "max_tokens": 800,
                },
                timeout=45,
            )

            if response.status_code == 200:
                data = response.json()
                content = data['choices'][0]['message']['content']
                variations = []
                for line in content.strip().split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit()):
                        # Remove numbering like "1. " or "1) "
                        parts = line.split('. ', 1) if '. ' in line else line.split(') ', 1)
                        text = parts[1] if len(parts) > 1 else line
                        text = text.strip().strip('"').strip("'")
                        if len(text) > 5:
                            variations.append(text)

                # Ensure the literal text is included
                if literal not in variations:
                    variations.insert(0, literal)

                results[key] = variations[:8]  # Max 8
                processed += 1
                print(f"  [{processed}/{len(texts_info)}] {key}: {len(variations[:8])} variations")
            else:
                print(f"  [{processed}/{len(texts_info)}] {key}: API error {response.status_code}")
                results[key] = [literal]  # Fallback to literal only
                processed += 1

        except Exception as e:
            print(f"  [{processed}/{len(texts_info)}] {key}: Error: {e}")
            results[key] = [literal]
            processed += 1

        time.sleep(0.3)  # Rate limiting

    return results


# ─── STEP 3: TOKENIZE BVHs ───
def tokenize_bvhs(file_list, max_files=None):
    """Tokenize selected BVH files using MoConVQ encoder."""
    print(f"\n{'=' * 60}")
    print(f"[3/4] Tokenizing BVH files...")

    if max_files:
        file_list = file_list[:max_files]

    os.makedirs(BVH_OUT_DIR, exist_ok=True)

    # Extract BVH files from zip
    print(f"  Extracting {len(file_list)} BVH files...")
    with zipfile.ZipFile(BANDAI_ZIP, 'r') as zf:
        for i, f in enumerate(file_list):
            out_path = os.path.join(BVH_OUT_DIR, f['basename'])
            if not os.path.exists(out_path):
                zf.extract(f['zip_path'], BVH_OUT_DIR)
                # Move from nested path to flat
                extracted = os.path.join(BVH_OUT_DIR, f['zip_path'])
                if os.path.exists(extracted) and extracted != out_path:
                    os.rename(extracted, out_path)
            if (i + 1) % 50 == 0:
                print(f"    Extracted {i+1}/{len(file_list)}...")

    # Clean up nested directories
    nested = os.path.join(BVH_OUT_DIR, 'Bandai-Namco-Research-Motiondataset-master')
    if os.path.exists(nested):
        import shutil
        shutil.rmtree(nested)

    # Tokenize using MoConVQ
    print(f"  Loading MoConVQ models...")
    from MoConVQCore.Env.vclode_track_env import VCLODETrackEnv
    from MoConVQCore.Model.MoConVQ import MoConVQ
    from MoConVQCore.Utils.misc import load_yaml
    from MoConVQCore.Utils.motion_dataset import MotionDataSet
    import MoConVQCore.Utils.pytorch_utils as ptu
    import numpy as np
    import torch

    # Build agent
    ptu.init_gpu(True, gpu_id=0)
    env = VCLODETrackEnv(
        scene_fname=os.path.join(BASE_DIR, 'Data', 'Misc', 'world.json'),
        fps=20,
    )
    agent = MoConVQ(323, 12, 57, 120, env, training=False)
    agent.simple_load(os.path.join(BASE_DIR, 'moconvq_base.data'), strict=True)
    agent.eval()
    agent.posterior.limit = False
    torch.set_grad_enabled(False)

    print(f"  Tokenizing {len(file_list)} BVH files...")
    tokens_dict = {}
    failed = []

    for i, f in enumerate(file_list):
        bvh_path = os.path.join(BVH_OUT_DIR, f['basename'])
        key = os.path.splitext(f['basename'])[0]

        try:
            motion_data = MotionDataSet(20)
            motion_data.add_bvh_with_character(bvh_path, env.sim_character)

            info = agent.encode_seq_all(None, motion_data.observation)
            indices = info['indexs']  # shape: (seq_len, num_vq_layers)

            tokens_dict[key] = indices.cpu().numpy() if hasattr(indices, 'cpu') else indices

            if (i + 1) % 20 == 0:
                print(f"    Tokenized {i+1}/{len(file_list)}... ({len(failed)} failed)")
        except Exception as e:
            failed.append((key, str(e)[:100]))
            if len(failed) <= 5:
                print(f"    ✗ {key}: {str(e)[:120]}")

    print(f"  Tokenization complete: {len(tokens_dict)} success, {len(failed)} failed")

    # Save tokens
    np.savez_compressed(TOKENS_OUT, **tokens_dict)
    print(f"  Tokens saved to: {TOKENS_OUT}")

    return tokens_dict, failed


# ─── STEP 4: ASSEMBLE DATASET ───
def assemble_dataset(groups, augmented_texts, tokens_dict):
    """Combine texts, tokens into final training dataset."""
    print(f"\n{'=' * 60}")
    print(f"[4/4] Assembling final dataset...")

    train_entries = []
    val_entries = []

    for (action, style), file_infos in groups.items():
        key = f"{action}_{style}"
        texts = augmented_texts.get(key, [generate_literal_text(action, style)])

        for fi in file_infos:
            bvh_key = os.path.splitext(fi['basename'])[0]
            if bvh_key not in tokens_dict:
                continue

            token_info = {
                'bvh_file': fi['basename'],
                'token_key': bvh_key,
                'n_frames': int(tokens_dict[bvh_key].shape[0]),
            }

            for text in texts:
                entry = {
                    'text': text,
                    'action': action,
                    'style': style,
                    **token_info,
                }
                # 80/20 train/val split by take number
                if fi['take'] <= 8:
                    train_entries.append(entry)
                else:
                    val_entries.append(entry)

    dataset = {
        'metadata': {
            'source': 'Bandai Namco Research Motion Dataset',
            'description': 'Style-labeled mocap BVH files with DeepSeek-augmented text descriptions',
            'total_train': len(train_entries),
            'total_val': len(val_entries),
            'total_unique_bvh': len(tokens_dict),
            'actions': list(set(e['action'] for e in train_entries)),
            'styles': list(set(e['style'] for e in train_entries)),
        },
        'train': train_entries,
        'val': val_entries,
    }

    with open(DATASET_OUT, 'w') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"  Dataset saved to: {DATASET_OUT}")
    print(f"  Train entries: {len(train_entries)}")
    print(f"  Val entries: {len(val_entries)}")
    print(f"  Unique BVH files tokenized: {len(tokens_dict)}")
    print(f"  Unique text descriptions: {len(set(e['text'] for e in train_entries))}")

    return dataset


# ─── MAIN ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, default=None, help='DeepSeek API key')
    parser.add_argument('--skip-augment', action='store_true', help='Skip text augmentation')
    parser.add_argument('--skip-tokenize', action='store_true', help='Skip tokenization')
    parser.add_argument('--max-files', type=int, default=500, help='Max BVH files to tokenize')
    parser.add_argument('--actions', type=str, nargs='*', default=None, help='Filter by actions')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Catalog
    all_files, groups = catalog_files()

    # Filter: only keep actions with enough samples
    if args.actions:
        groups = {k: v for k, v in groups.items() if k[0] in args.actions}
    else:
        # Default: focus on locomotion-relevant actions
        priority_actions = {'walk', 'run', 'walking', 'running', 'dash'}
        groups = {k: v for k, v in groups.items() if k[0] in priority_actions}

    print(f"\n  Filtered to {len(groups)} (action, style) pairs")

    # Prepare text info
    texts_info = {}
    for (action, style), file_infos in groups.items():
        literal = generate_literal_text(action, style)
        texts_info[(action, style)] = {'literal': literal, 'files': file_infos}

    # Step 2: Text augmentation
    if args.skip_augment or not args.api_key:
        if not args.api_key:
            print("\n[2/4] No API key, using literal texts only")
        augmented = {(a, s): [info['literal']] for (a, s), info in texts_info.items()}
    else:
        augmented = augment_texts_with_deepseek(texts_info, args.api_key)
        # Save augmented texts (augmented has string keys like "walk_proud")
        aug_path = os.path.join(OUTPUT_DIR, 'augmented_texts.json')
        with open(aug_path, 'w') as f:
            json.dump(augmented, f, indent=2, ensure_ascii=False)
        print(f"  Augmented texts saved to: {aug_path}")
        # Also save human-readable texts list
        texts_path = os.path.join(OUTPUT_DIR, 'all_texts.json')
        all_texts = {}
        for key, texts in augmented.items():
            all_texts[key] = texts
        with open(texts_path, 'w') as f:
            json.dump(all_texts, f, indent=2, ensure_ascii=False)

    # Step 3: Tokenize
    if args.skip_tokenize:
        print("\n[3/4] Skipping tokenization")
        # Try to load existing tokens
        if os.path.exists(TOKENS_OUT):
            tokens_dict = dict(np.load(TOKENS_OUT, allow_pickle=True))
            tokens_dict = {k: v for k, v in tokens_dict.items()}
            print(f"  Loaded {len(tokens_dict)} existing tokens")
        else:
            tokens_dict = {}
    else:
        # Flatten file list
        file_list = []
        for (action, style), file_infos in groups.items():
            file_list.extend(file_infos)
        file_list.sort(key=lambda x: x['basename'])
        tokens_dict, failed = tokenize_bvhs(file_list, max_files=args.max_files)

    # Step 4: Assemble
    if tokens_dict:
        dataset = assemble_dataset(groups, augmented, tokens_dict)
        print(f"\n{'=' * 60}")
        print("Done! Dataset ready for fine-tuning.")
        print(f"  Output: {DATASET_OUT}")
        print(f"  Tokens: {TOKENS_OUT}")
    else:
        print("\nNo tokens available. Run without --skip-tokenize first.")


if __name__ == '__main__':
    main()
