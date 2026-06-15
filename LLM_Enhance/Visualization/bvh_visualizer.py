"""
BVH skeleton visualization — render motion capture data as rotating 3D GIF.
Uses matplotlib 3D axes + PillowWriter (no ffmpeg required).

Usage:
  python bvh_visualizer.py input.bvh -o output.gif
  python bvh_visualizer.py input.bvh --view fixed --fps-ratio 2

Or programmatically:
  from bvh_visualizer import visualize_bvh
  visualize_bvh("motion.bvh", "output.gif")
"""

import argparse
import numpy as np
import os

_DEFAULT_FPS_RATIO = 4       # 120 → 30 FPS
_DEFAULT_ELEVATION = 25
_DEFAULT_ROTATION_SPAN = 270
_DEFAULT_DPI = 100
_DEFAULT_FIGSIZE = (8, 8)
_PADDING_RATIO = 0.1
_END_SITE_INDICES = {4, 9, 14, 19, 24}


def _get_joint_color(names):
    """Replicate pymotionlib.utils.get_joint_color logic: left→red, right→blue, other→orange."""
    matches = (
        ('l', 'r'),
        ('L', 'R'),
        ('left', 'right'),
        ('Left', 'Right'),
        ('LEFT', 'RIGHT'),
    )

    def check(n, i):
        for m in matches:
            prefix = m[i]
            counterpart = m[1 - i]
            if n[:len(prefix)] == prefix and counterpart + n[len(prefix):] in names:
                return True
            if n[-len(prefix):] == prefix and n[:-len(prefix)] + counterpart in names:
                return True
        return False

    colors = []
    for n in names:
        if check(n, 0):
            colors.append('#E74C3C')  # red for left
        elif check(n, 1):
            colors.append('#3498DB')  # blue for right
        else:
            colors.append('#F39C12')  # orange for center
    return colors


def load_bvh(bvh_path):
    """Load a BVH file via VclSimuBackend. Returns MotionData."""
    try:
        import VclSimuBackend
    except ImportError:
        raise ImportError(
            "VclSimuBackend is not importable. "
            "Activate the moconvq conda environment first."
        )
    return VclSimuBackend.pymotionlib.BVHLoader.load(bvh_path)


def _compute_static_bounds(positions):
    """Bounding box in root-relative space (centered around origin for X/Z)."""
    root_pos = positions[:, 0:1, :]               # (N, 1, 3)
    local = positions.copy()
    local[:, :, 0] = local[:, :, 0] - root_pos[:, 0, 0:1]
    local[:, :, 2] = local[:, :, 2] - root_pos[:, 0, 2:3]
    lo = np.min(local, axis=(0, 1))
    hi = np.max(local, axis=(0, 1))
    pad = (hi - lo) * _PADDING_RATIO
    lo -= pad
    hi += pad
    return lo, hi


def _plot_skeleton(ax, positions, parents, colors, end_sites):
    """Draw one frame of the skeleton on the given 3D axes."""
    artists = []
    n_joints = len(parents)

    # Remap coordinates: matplot X=BVH X, matplot Z=BVH Y, matplot Y=-BVH Z
    xs = positions[:, 0]
    zs = positions[:, 1]
    ys = -positions[:, 2]

    for j in range(n_joints):
        p = parents[j]
        if p < 0:
            continue
        color = 'k' if j in end_sites else colors[j]
        marker = 'o' if j in end_sites else 'x'
        ms = 3 if j in end_sites else 5
        (line,) = ax.plot(
            [xs[p], xs[j]],
            [ys[p], ys[j]],
            [zs[p], zs[j]],
            color=color,
            marker=marker,
            markersize=ms,
            linewidth=1.5,
        )
        artists.append(line)
    return artists


def _build_rotating_angles(num_frames, span):
    """Linearly interpolated azimuth angles for rotating camera."""
    return np.linspace(0, span, num_frames)


def visualize_bvh(bvh_path, output_path, **kwargs):
    """Load a BVH file and render it as a GIF animation.

    Parameters
    ----------
    bvh_path : str
        Path to input .bvh file.
    output_path : str
        Path for output .gif file.
    output_fps : int
        Target FPS of the output GIF (default 30).
    fps_ratio : int
        Downsample factor: output_fps = source_fps / fps_ratio (default 4).
    view_mode : str
        'rotate' (default) or 'fixed'.
    elevation : float
        Camera elevation in degrees (default 25).
    rotation_span : float
        Total azimuth rotation in degrees for 'rotate' mode (default 270).
    follow_root : bool
        Whether camera follows root joint horizontal motion (default True).
    dpi : int
        Output resolution (default 100).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    motion = load_bvh(bvh_path)

    positions = np.array(motion._joint_position, dtype=np.float64)
    parents = list(motion._skeleton_joint_parents)
    joint_names = list(motion._skeleton_joints)
    end_sites = set(motion.end_sites) if hasattr(motion, 'end_sites') and motion.end_sites else _END_SITE_INDICES
    source_fps = int(motion._fps)

    num_frames, num_joints, _ = positions.shape

    # Downsample
    fps_ratio = kwargs.get('fps_ratio', _DEFAULT_FPS_RATIO)
    output_fps = kwargs.get('output_fps', source_fps // fps_ratio)
    if fps_ratio != _DEFAULT_FPS_RATIO or 'output_fps' not in kwargs:
        fps_ratio = max(1, source_fps // output_fps)

    indices = list(range(0, num_frames, fps_ratio))
    positions = positions[indices]
    num_out_frames = len(indices)

    # Precompute
    colors = _get_joint_color(joint_names)
    lo, hi = _compute_static_bounds(positions)
    center_xz = np.array([0.0, 0.0])
    view_mode = kwargs.get('view_mode', 'fixed')
    follow_root = kwargs.get('follow_root', True)

    if view_mode == 'rotate':
        azimuths = _build_rotating_angles(num_out_frames, kwargs.get('rotation_span', _DEFAULT_ROTATION_SPAN))
    else:
        azimuths = np.full(num_out_frames, 45.0)

    elevation = kwargs.get('elevation', _DEFAULT_ELEVATION)
    dpi = kwargs.get('dpi', _DEFAULT_DPI)

    fig = plt.figure(figsize=_DEFAULT_FIGSIZE, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # Override y/z labels since we remapped
    ax.set_xlabel('Forward')
    ax.set_ylabel('Lateral')
    ax.set_zlabel('Up')

    ax.set_box_aspect((hi[0] - lo[0], hi[2] - lo[2], hi[1] - lo[1]))

    def update(frame_idx):
        ax.cla()
        pos = positions[frame_idx]

        if follow_root:
            center_xz[0] = pos[0, 0]
            center_xz[1] = pos[0, 2]

        _plot_skeleton(ax, pos, parents, colors, end_sites)

        azim = azimuths[frame_idx]
        ax.view_init(elev=elevation, azim=azim)

        ax.set_xlim(lo[0] + center_xz[0], hi[0] + center_xz[0])
        ax.set_ylim(-(hi[2] + center_xz[1]), -(lo[2] + center_xz[1]))
        ax.set_zlim(lo[1], hi[1])

        ax.set_xlabel('Forward')
        ax.set_ylabel('Lateral')
        ax.set_zlabel('Up')
        ax.set_title(f'Frame {frame_idx * fps_ratio}/{num_frames}  |  {source_fps} fps')

    ani = animation.FuncAnimation(fig, update, frames=num_out_frames, interval=1000 / output_fps, blit=False)

    writer = animation.PillowWriter(fps=output_fps)
    ani.save(output_path, writer=writer)
    plt.close(fig)
    print(f'Saved: {output_path}  ({num_out_frames} frames, {output_fps} fps)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Render BVH skeleton animation to GIF')
    parser.add_argument('bvh_path', help='Input BVH file')
    parser.add_argument('-o', '--output', default=None, help='Output GIF path (default: <input_stem>.gif)')
    parser.add_argument('--fps-ratio', type=int, default=_DEFAULT_FPS_RATIO,
                        help=f'Downsample factor (default: {_DEFAULT_FPS_RATIO})')
    parser.add_argument('--view', choices=('rotate', 'fixed'), default='fixed',
                        help='Camera mode (default: rotate)')
    parser.add_argument('--elevation', type=float, default=_DEFAULT_ELEVATION,
                        help=f'Camera elevation degrees (default: {_DEFAULT_ELEVATION})')
    parser.add_argument('--rotation-span', type=float, default=_DEFAULT_ROTATION_SPAN,
                        help=f'Rotation span in degrees (default: {_DEFAULT_ROTATION_SPAN})')
    parser.add_argument('--no-follow-root', action='store_true',
                        help='Disable camera following root motion')
    parser.add_argument('--dpi', type=int, default=_DEFAULT_DPI,
                        help=f'Output DPI (default: {_DEFAULT_DPI})')

    args = parser.parse_args()

    output = args.output
    if output is None:
        output = os.path.splitext(args.bvh_path)[0] + '.gif'

    visualize_bvh(
        args.bvh_path,
        output,
        fps_ratio=args.fps_ratio,
        view_mode=args.view,
        elevation=args.elevation,
        rotation_span=args.rotation_span,
        follow_root=not args.no_follow_root,
        dpi=args.dpi,
    )
