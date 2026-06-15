"""Multi-angle BVH visualization — renders the same motion from 4 camera angles
simultaneously (front, side, top, perspective), giving a true 3D view of the
ODE physics simulation output.

Usage:
  python Visualization/multi_angle_viewer.py <input.bvh> [-o output.gif]
"""

import argparse
import numpy as np
import os


def visualize_multi_angle(bvh_path, output_path, output_fps=24, dpi=120):
    """Render BVH from 4 camera angles in a 2x2 grid.

    Angles: Front (0°), Side (90°), Perspective (45°/30°), Top (90° elevation)
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # Load BVH via VclSimuBackend
    import VclSimuBackend

    motion = VclSimuBackend.pymotionlib.BVHLoader.load(bvh_path)

    positions = np.array(motion._joint_position, dtype=np.float64)
    parents = list(motion._skeleton_joint_parents)
    joint_names = list(motion._skeleton_joints)
    end_sites = set(motion.end_sites) if hasattr(motion, 'end_sites') and motion.end_sites else {4, 9, 14, 19, 24}
    source_fps = int(motion._fps)

    num_frames, num_joints, _ = positions.shape

    # Downsample to target FPS
    step = max(1, source_fps // output_fps)
    indices = list(range(0, num_frames, step))
    positions = positions[indices]
    num_out_frames = len(indices)

    # Joint colors: left=red, right=blue, other=orange
    colors = _get_joint_color(joint_names)

    # Compute static bounds (root-relative)
    root_pos = positions[:, 0:1, :]
    local = positions.copy()
    local[:, :, 0] -= root_pos[:, 0, 0:1]
    local[:, :, 2] -= root_pos[:, 0, 2:3]
    lo = np.min(local, axis=(0, 1)) - 0.15
    hi = np.max(local, axis=(0, 1)) + 0.15

    # 4 camera configurations: (elevation, azimuth, title)
    cameras = [
        (15, 0,   'Front View'),
        (15, 90,  'Side View'),
        (25, 45,  'Perspective View'),
        (85, 0,   'Top View'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=dpi,
                              subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    fig.suptitle(f'MoConVQ Physics Simulation — {os.path.basename(bvh_path)}',
                 fontsize=13, fontweight='bold', y=0.98)

    def draw_skeleton(ax, pos, frame_idx):
        ax.cla()
        xs, ys, zs = pos[:, 0], -pos[:, 2], pos[:, 1]  # remap: matplot Z=up

        for j in range(num_joints):
            p = parents[j]
            if p < 0:
                continue
            c = 'k' if j in end_sites else colors[j]
            m = 'o' if j in end_sites else 'x'
            s = 3 if j in end_sites else 5
            ax.plot([xs[p], xs[j]], [ys[p], ys[j]], [zs[p], zs[j]],
                    color=c, marker=m, markersize=s, linewidth=1.5)

        rx, ry, rz = pos[0, 0], -pos[0, 2], pos[0, 1]
        ax.set_xlim(rx + lo[0], rx + hi[0])
        ax.set_ylim(ry + (-hi[2]), ry + (-lo[2]))
        ax.set_zlim(rz + lo[1], rz + hi[1])

        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z (Up)')

    def update(frame_idx):
        pos = positions[frame_idx]
        real_frame = frame_idx * step
        for ax, (elev, azim, title) in zip(axes, cameras):
            draw_skeleton(ax, pos, real_frame)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(title, fontsize=11, fontweight='bold')
        fig.suptitle(f'MoConVQ Physics Simulation — {os.path.basename(bvh_path)}  |  '
                     f'Frame {real_frame}/{num_frames}  ({source_fps} fps)',
                     fontsize=13, fontweight='bold', y=0.98)

    ani = animation.FuncAnimation(fig, update, frames=num_out_frames,
                                   interval=1000 / output_fps, blit=False)
    writer = animation.PillowWriter(fps=output_fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f'Saved multi-angle view: {output_path}  ({num_out_frames} frames, {output_fps} fps)')


def _get_joint_color(names):
    """left→red, right→blue, other→orange."""
    matches = (('l', 'r'), ('L', 'R'), ('left', 'right'),
               ('Left', 'Right'), ('LEFT', 'RIGHT'))

    def check(n, i):
        for m in matches:
            prefix, counter = m[i], m[1 - i]
            if n[:len(prefix)] == prefix and counter + n[len(prefix):] in names:
                return True
            if n[-len(prefix):] == prefix and n[:-len(prefix)] + counter in names:
                return True
        return False

    colors = []
    for n in names:
        if check(n, 0):   colors.append('#E74C3C')
        elif check(n, 1): colors.append('#3498DB')
        else:             colors.append('#F39C12')
    return colors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-angle BVH → GIF')
    parser.add_argument('bvh_path', help='Input BVH file')
    parser.add_argument('-o', '--output', default=None, help='Output GIF path')
    parser.add_argument('--fps', type=int, default=24, help='Output FPS (default 24)')
    parser.add_argument('--dpi', type=int, default=120, help='DPI (default 120)')
    args = parser.parse_args()

    output = args.output or os.path.splitext(args.bvh_path)[0] + '_multi.gif'
    visualize_multi_angle(args.bvh_path, output, output_fps=args.fps, dpi=args.dpi)
