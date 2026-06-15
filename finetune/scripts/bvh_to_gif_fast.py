#!/usr/bin/env python3
"""
Fast BVH→GIF renderer. Features:
- Dynamically reads skeleton from BVH (always correct)
- 120fps→20fps subsampling (preserves original speed)
- PIL direct GIF output (no ffmpeg)
- Parallel multi-file rendering

Usage:
  python Script/bvh_to_gif_fast.py in.bvh [out.gif]
  python Script/bvh_to_gif_fast.py --dir bvh_folder/ --out gif_folder/ --parallel 15
"""

import sys, os, warnings, argparse, json
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

# ─── BVH reading (uses pymotionlib for correctness) ───
def load_bvh(bvh_path):
    """Load BVH using pymotionlib, return (positions, joint_names, parents, fps)."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ModifyODESrc'))
    import VclSimuBackend
    loader = VclSimuBackend.pymotionlib.BVHLoader()
    motion = loader.load(bvh_path)
    nf = motion.num_frames
    pos = motion.joint_position
    if pos.ndim == 2:
        pos = pos.reshape(nf, -1, 3)
    return (pos,                     # (frames, joints, 3) world positions
            motion.joint_names,       # list of joint names
            motion.joint_parents_idx, # parent index per joint
            120.0)                    # MoConVQ always uses 120fps

# ─── Coordinate transform ───
def _to_plot(pos_bvh):
    return np.array([pos_bvh[0], pos_bvh[2], pos_bvh[1]])

# ─── Joint coloring (heuristic based on name) ───
def get_joint_color(name):
    name_l = name.lower()
    if 'head' in name_l or 'torso_head' in name_l: return 'gold'
    if 'clavicle' in name_l or 'rshoulder' in name_l or 'lshoulder' in name_l: return None  # use bone color
    if 'relbow' in name_l: return 'red'
    if 'lelbow' in name_l: return 'blue'
    if 'rwrist' in name_l: return 'darkred'
    if 'lwrist' in name_l: return 'darkblue'
    if 'rhip' in name_l or 'rknee' in name_l or 'rankle' in name_l or 'rtoe' in name_l: return 'darkorange'
    if 'lhip' in name_l or 'lknee' in name_l or 'lankle' in name_l or 'ltoe' in name_l: return 'cyan'
    if 'pelvis' in name_l or 'lowerback' in name_l or 'torso' in name_l or 'root' in name_l: return 'forestgreen'
    return 'gray'

def get_bone_color(child_name):
    """Color bone by which side of body."""
    name_l = child_name.lower()
    if 'rhip' in name_l or 'rknee' in name_l or 'rankle' in name_l or 'rtoe' in name_l: return 'darkorange'
    if 'lhip' in name_l or 'lknee' in name_l or 'lankle' in name_l or 'ltoe' in name_l: return 'cyan'
    if 'rshoulder' in name_l or 'relbow' in name_l or 'rwrist' in name_l or 'rclavicle' in name_l or 'rtorso' in name_l: return 'red'
    if 'lshoulder' in name_l or 'lelbow' in name_l or 'lwrist' in name_l or 'lclavicle' in name_l or 'ltorso' in name_l: return 'blue'
    if 'head' in name_l: return 'gold'
    return 'forestgreen'


def render_one_gif(bvh_path, output_path, target_fps=20, dpi=80, title=None):
    """Render a single BVH to GIF using the skeleton from the BVH file."""
    positions, joint_names, parents, source_fps = load_bvh(bvh_path)
    total_frames = positions.shape[0]

    # Correct subsampling: every N frames for 120→20fps
    step = max(1, int(source_fps / target_fps))
    indices = list(range(0, total_frames, step))
    n_render = len(indices)

    # Build bone connections from actual BVH hierarchy
    bone_list = []
    for i, name in enumerate(joint_names):
        p = parents[i]
        if p >= 0 and 'end' not in name.lower() and 'end' not in joint_names[p].lower():
            bone_list.append((joint_names[p], name))

    # Transform all positions to plot coordinates
    world_positions = []
    for fi in indices:
        plot_pos = {}
        for i, name in enumerate(joint_names):
            plot_pos[name] = _to_plot(positions[fi, i])
        world_positions.append(plot_pos)

    # Compute global bounds
    all_x, all_y, all_z = [], [], []
    for fp in world_positions:
        for p in fp.values():
            all_x.append(p[0]); all_y.append(p[1]); all_z.append(p[2])

    gx_min, gx_max = min(all_x), max(all_x)
    gy_min, gy_max = min(all_y), max(all_y)
    gz_min = min(all_z)

    margin = max(gx_max - gx_min, gy_max - gy_min, 0.5) * 0.1
    gx_min -= margin; gx_max += margin
    gy_min -= margin; gy_max += margin

    # Reusable figure
    fig = plt.figure(figsize=(8, 8), dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    frames_pil = []
    for i, plot_pos in enumerate(world_positions):
        ax.clear()
        ax.set_xlim(gx_min, gx_max); ax.set_ylim(gy_min, gy_max)
        ax.set_zlim(gz_min - 0.02, gz_min + 2.2)
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.view_init(elev=25, azim=-60)
        if title: ax.set_title(title, fontsize=10)

        # Ground + shadow
        gx_g = np.linspace(gx_min-0.5, gx_max+0.5, 4)
        gy_g = np.linspace(gy_min-0.5, gy_max+0.5, 4)
        GX, GY = np.meshgrid(gx_g, gy_g)
        ax.plot_surface(GX, GY, np.full_like(GX, gz_min-0.02), color='lightgray', alpha=0.2)
        for parent, child in bone_list:
            pp, cp = plot_pos.get(parent), plot_pos.get(child)
            if pp is not None and cp is not None:
                ax.plot([pp[0], cp[0]], [pp[1], cp[1]], [gz_min-0.01]*2,
                       color='gray', alpha=0.12, linewidth=0.8)

        # Bones
        for parent, child in bone_list:
            pp, cp = plot_pos.get(parent), plot_pos.get(child)
            if pp is not None and cp is not None:
                ax.plot([pp[0], cp[0]], [pp[1], cp[1]], [pp[2], cp[2]],
                       color=get_bone_color(child), linewidth=3.5, solid_capstyle='round')

        # Joints
        for name, pos in plot_pos.items():
            if 'end' not in name.lower() and pos is not None:
                ax.scatter(pos[0], pos[1], pos[2], c=get_joint_color(name),
                          s=22, edgecolors='black', linewidths=0.4)

        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=80, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf).convert('RGBA')
        img = img.resize((img.width//2, img.height//2), Image.LANCZOS)
        frames_pil.append(img)
        buf.close()

    plt.close(fig)

    # Write GIF
    duration = int(1000 / target_fps)
    frames_pil[0].save(output_path, format='GIF', save_all=True,
                       append_images=frames_pil[1:], duration=duration,
                       loop=0, optimize=True, disposal=2)

    return len(frames_pil), total_frames


def render_worker(args_tuple):
    """Worker for parallel rendering."""
    bvh_path, output_path, fps, dpi, title = args_tuple
    try:
        nf, total = render_one_gif(bvh_path, output_path, fps, dpi, title)
        return (os.path.basename(bvh_path), True, f"{nf}f/{total}f", None)
    except Exception as e:
        return (os.path.basename(bvh_path), False, None, str(e)[:120])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', help='Input BVH file')
    parser.add_argument('output', nargs='?', default=None, help='Output GIF file')
    parser.add_argument('--dir', type=str, help='Input directory of BVH files')
    parser.add_argument('--out', type=str, default=None, help='Output directory')
    parser.add_argument('--fps', type=int, default=20)
    parser.add_argument('--dpi', type=int, default=80)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel workers')
    args = parser.parse_args()

    if args.dir:
        bvh_dir = args.dir
        out_dir = args.out or os.path.join(os.path.dirname(bvh_dir), 'gif')
        os.makedirs(out_dir, exist_ok=True)

        bvh_files = sorted([f for f in os.listdir(bvh_dir) if f.endswith('.bvh')])
        print(f"Rendering {len(bvh_files)} BVH files → {out_dir}/")
        print(f"Parallel workers: {args.parallel}")

        tasks = []
        for f in bvh_files:
            tasks.append((
                os.path.join(bvh_dir, f),
                os.path.join(out_dir, f.replace('.bvh', '.gif')),
                args.fps, args.dpi,
                f.replace('.bvh', '')  # title from filename
            ))

        if args.parallel > 1:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            with ProcessPoolExecutor(max_workers=args.parallel) as pool:
                futures = {pool.submit(render_worker, t): t for t in tasks}
                done = 0
                for future in as_completed(futures):
                    name, ok, info, err = future.result()
                    done += 1
                    if ok:
                        print(f"  [{done}/{len(tasks)}] ✓ {name} ({info})")
                    else:
                        print(f"  [{done}/{len(tasks)}] ✗ {name}: {err}")
        else:
            for i, t in enumerate(tasks):
                name, ok, info, err = render_worker(t)
                if ok:
                    print(f"  [{i+1}/{len(tasks)}] ✓ {name} ({info})")
                else:
                    print(f"  [{i+1}/{len(tasks)}] ✗ {name}: {err}")

        print(f"\nDone! {len(tasks)} GIFs in {out_dir}/")
    elif args.input:
        out = args.output or os.path.splitext(args.input)[0] + '.gif'
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        nf, total = render_one_gif(args.input, out, args.fps, args.dpi, args.title)
        print(f"Saved: {out} ({nf}f, {nf/args.fps:.1f}s)")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
