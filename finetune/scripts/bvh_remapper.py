#!/usr/bin/env python3
"""
BVH skeleton remapper: converts external mocap BVH to MoConVQ ODE-compatible format.

Approach:
1. Read source BVH via pymotionlib → extract world-space joint positions
2. Map source joint names → ODE joint names via configurable mapping
3. Build new BVH with ODE skeleton hierarchy + mapped position data
4. Save MoConVQ-compatible BVH

Usage:
  python Script/bvh_remapper.py --input in.bvh --output out.bvh --skeleton cmu
  python Script/bvh_remapper.py --input-dir bvh_folder/ --output-dir ode_bvh/ --skeleton bandai
"""

import sys, os, re, argparse, json
import numpy as np
from collections import OrderedDict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, 'ModifyODESrc'))

# ─── ODE Skeleton Definition ───
# (joint_name, parent_index, offset) - matches MoConVQ's ODE character
ODE_SKELETON = [
    # name,              parent_idx, offset(x, y, z)
    ('pelvis_lowerback',  -1, (0.000,  0.093605,  0.000)),
    ('lowerback_torso',    0, (0.000,  0.100000,  0.000)),
    ('rHip',               0, (-0.087, -0.060000,  0.000)),
    ('lHip',               0, (0.087,  -0.060000,  0.000)),
    ('rKnee',              2, (0.000,  -0.350000,  0.000)),
    ('lKnee',              3, (0.000,  -0.350000,  0.000)),
    ('rAnkle',             4, (0.000,  -0.350000,  0.000)),
    ('lAnkle',             5, (0.000,  -0.350000,  0.000)),
    ('rToeJoint',          6, (0.000,  -0.080000,  0.070)),
    ('lToeJoint',          7, (0.000,  -0.080000,  0.070)),
    ('torso_head',         1, (0.000,  0.282350,  0.000)),
    ('rTorso_Clavicle',    1, (-0.001,  0.157500,  0.000)),
    ('lTorso_Clavicle',    1, (0.001,   0.157500,  0.000)),
    ('rShoulder',         11, (-0.117647, 0.000000, 0.000)),
    ('lShoulder',         12, (0.117647,  0.000000, 0.000)),
    ('rElbow',            13, (-0.245000, 0.000000, 0.000)),
    ('lElbow',            14, (0.245000,  0.000000, 0.000)),
    ('rWrist',            15, (-0.240000, 0.000000, 0.000)),
    ('lWrist',            16, (0.240000,  0.000000, 0.000)),
]

ODE_END_SITES = [
    (10, (0.000, 0.192650, 0.000)),   # torso_head → head tip
    (8,  (0.000, 0.000000, 0.070)),   # rToeJoint → toe tip
    (9,  (0.000, 0.000000, 0.070)),   # lToeJoint → toe tip
    (17, (-0.116353, -0.002500, 0.000)), # rWrist → hand tip
    (18, (0.116353,  -0.002500, 0.000)), # lWrist → hand tip
]


# ─── Joint Name Mappings ───
# Maps source BVH joint names → ODE joint names

CMU_MAPPING = {
    'Hips': 'pelvis_lowerback',
    'LowerBack': 'lowerback_torso',
    'Spine': 'lowerback_torso',
    'Spine1': 'torso_head',
    'Neck': 'torso_head',
    'Neck1': 'torso_head',
    'Head': 'torso_head',
    'RHipJoint': 'rHip',
    'RightUpLeg': 'rHip',
    'RightLeg': 'rKnee',
    'RightFoot': 'rAnkle',
    'RightToeBase': 'rToeJoint',
    'LHipJoint': 'lHip',
    'LeftUpLeg': 'lHip',
    'LeftLeg': 'lKnee',
    'LeftFoot': 'lAnkle',
    'LeftToeBase': 'lToeJoint',
    'RightShoulder': 'rTorso_Clavicle',
    'RightArm': 'rShoulder',
    'RightForeArm': 'rElbow',
    'RightHand': 'rWrist',
    'LeftShoulder': 'lTorso_Clavicle',
    'LeftArm': 'lShoulder',
    'LeftForeArm': 'lElbow',
    'LeftHand': 'lWrist',
}

LAFAN1_MAPPING = {
    'Hips': 'pelvis_lowerback',
    'Spine': 'lowerback_torso',
    'Spine1': 'lowerback_torso',
    'Spine2': 'torso_head',
    'Neck': 'torso_head',
    'Head': 'torso_head',
    'RightUpLeg': 'rHip',
    'RightLeg': 'rKnee',
    'RightFoot': 'rAnkle',
    'RightToe': 'rToeJoint',
    'LeftUpLeg': 'lHip',
    'LeftLeg': 'lKnee',
    'LeftFoot': 'lAnkle',
    'LeftToe': 'lToeJoint',
    'RightShoulder': 'rTorso_Clavicle',
    'RightArm': 'rShoulder',
    'RightForeArm': 'rElbow',
    'RightHand': 'rWrist',
    'LeftShoulder': 'lTorso_Clavicle',
    'LeftArm': 'lShoulder',
    'LeftForeArm': 'lElbow',
    'LeftHand': 'lWrist',
}

BANDAI_MAPPING = {
    'Hips': 'pelvis_lowerback',
    'Spine': 'lowerback_torso',
    'Chest': 'torso_head',
    'Neck': 'torso_head',
    'Head': 'torso_head',
    'UpperLeg_R': 'rHip',
    'LowerLeg_R': 'rKnee',
    'Foot_R': 'rAnkle',
    'Toes_R': 'rToeJoint',
    'UpperLeg_L': 'lHip',
    'LowerLeg_L': 'lKnee',
    'Foot_L': 'lAnkle',
    'Toes_L': 'lToeJoint',
    'Shoulder_R': 'rTorso_Clavicle',
    'UpperArm_R': 'rShoulder',
    'LowerArm_R': 'rElbow',
    'Hand_R': 'rWrist',
    'Shoulder_L': 'lTorso_Clavicle',
    'UpperArm_L': 'lShoulder',
    'LowerArm_L': 'lElbow',
    'Hand_L': 'lWrist',
}

SKELETON_MAPPINGS = {
    'cmu': CMU_MAPPING,
    'lafan1': LAFAN1_MAPPING,
    'bandai': BANDAI_MAPPING,
}


def extract_motion_data(bvh_path):
    """Extract positions and local rotations from a BVH file using pymotionlib."""
    import VclSimuBackend
    loader = VclSimuBackend.pymotionlib.BVHLoader()
    motion = loader.load(bvh_path)

    joint_names = motion.joint_names
    num_frames = motion.num_frames
    num_joints = len(joint_names)
    parents = motion.joint_parents_idx

    # World-space positions: (num_frames, num_joints, 3)
    pos = motion.joint_position
    if pos.ndim == 2:
        positions = pos.reshape(num_frames, num_joints, 3)
    else:
        positions = pos

    # Local rotations: (num_frames, num_joints, 4) quaternions [x,y,z,w]
    local_quats = motion.joint_orientation  # parent-relative

    return positions, local_quats, joint_names, parents, num_frames


def compute_world_rotations(local_quats, parents, num_frames, num_joints):
    """Compute world-space rotations via forward kinematics.
    world_rot[j] = world_rot[parent[j]] * local_quat[j]
    """
    from scipy.spatial.transform import Rotation as R

    world_rots = np.zeros((num_frames, num_joints, 4))
    for frame in range(num_frames):
        for j in range(num_joints):
            p = parents[j]
            local_r = R.from_quat(local_quats[frame, j, [0,1,2,3]])
            if p < 0:
                world_rots[frame, j] = local_r.as_quat()
            else:
                parent_r = R.from_quat(world_rots[frame, p, [0,1,2,3]])
                world_rots[frame, j] = (parent_r * local_r).as_quat()
    return world_rots


def build_ode_bvh(source_positions, source_quats, source_names, source_parents,
                  mapping, output_path, fps=120):
    """Build MoConVQ-compatible BVH with proper rotation handling.

    Uses world-space rotations from source BVH, maps to ODE joints,
    computes local rotations for ODE skeleton, writes BVH.
    """
    from scipy.spatial.transform import Rotation as R

    num_ode_joints = len(ODE_SKELETON)
    num_frames = source_positions.shape[0]
    num_src_joints = source_positions.shape[1]

    # Build source name → index
    src_name_to_idx = {name: i for i, name in enumerate(source_names)}

    # Compute world-space rotations for source
    src_world_rots = compute_world_rotations(source_quats, source_parents,
                                              num_frames, num_src_joints)

    # Map ODE joints to source joints
    ode_to_src = {}  # ode_idx → src_idx
    for ode_idx, (ode_name, _, _) in enumerate(ODE_SKELETON):
        for src_name, mapped_name in mapping.items():
            if mapped_name == ode_name and src_name in src_name_to_idx:
                ode_to_src[ode_idx] = src_name_to_idx[src_name]
                break

    # Extract ODE positions and world rotations from mapped source joints
    ode_positions = np.zeros((num_frames, num_ode_joints, 3))
    ode_world_rots = np.zeros((num_frames, num_ode_joints, 4))
    ode_world_rots[:, :] = [0, 0, 0, 1]  # Identity default

    for ode_idx, src_idx in ode_to_src.items():
        ode_positions[:, ode_idx, :] = source_positions[:, src_idx, :]
        ode_world_rots[:, ode_idx, :] = src_world_rots[:, src_idx, :]

    # Compute ODE local rotations: R_local = inv(R_parent_world) * R_joint_world
    ode_local_euler = np.zeros((num_frames, num_ode_joints, 3))  # ZYX degrees

    for frame in range(num_frames):
        for ode_idx, (ode_name, parent_idx, offset) in enumerate(ODE_SKELETON):
            joint_world_r = R.from_quat(ode_world_rots[frame, ode_idx, [0,1,2,3]])

            if parent_idx < 0:
                # Root: local = world (no parent)
                local_r = joint_world_r
            else:
                parent_world_r = R.from_quat(ode_world_rots[frame, parent_idx, [0,1,2,3]])
                local_r = parent_world_r.inv() * joint_world_r

            euler = local_r.as_euler('ZYX', degrees=True)
            ode_local_euler[frame, ode_idx] = euler

    # ─── Write BVH file ───
    with open(output_path, 'w') as f:
        f.write("HIERARCHY\n")
        _write_joint_hierarchy(f, 0, ODE_SKELETON, ODE_END_SITES, indent="")

        f.write(f"MOTION\nFrames: {num_frames}\nFrame Time: {1.0/fps:.6f}\n")

        for frame in range(num_frames):
            # Root position (x, y, z)
            rp = ode_positions[frame, 0]
            f.write(f"{rp[0]:.6f} {rp[1]:.6f} {rp[2]:.6f} ")
            # Root rotation ZYX (zrot, yrot, xrot)
            re = ode_local_euler[frame, 0]
            f.write(f"{re[0]:.6f} {re[1]:.6f} {re[2]:.6f} ")

            # Other joints (Zrotation Yrotation Xrotation)
            for ode_idx in range(1, num_ode_joints):
                e = ode_local_euler[frame, ode_idx]
                f.write(f"{e[0]:.6f} {e[1]:.6f} {e[2]:.6f} ")

            f.write("\n")

    return len(ode_to_src)


def _write_joint_hierarchy(f, idx, skeleton, end_sites, indent):
    """Recursively write BVH joint hierarchy."""
    name, parent_idx, offset = skeleton[idx]
    ox, oy, oz = offset

    if parent_idx < 0:  # Root
        f.write(f"ROOT {name}\n")
        f.write("{\n")
        f.write(f"  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}\n")
        f.write("  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation\n")
    else:
        f.write(f"{indent}JOINT {name}\n")
        f.write(f"{indent}{{\n")
        f.write(f"{indent}  OFFSET {ox:.6f} {oy:.6f} {oz:.6f}\n")
        f.write(f"{indent}  CHANNELS 3 Zrotation Yrotation Xrotation\n")

    # Write children
    children = [i for i, (_, p, _) in enumerate(skeleton) if p == idx]
    for child_idx in children:
        _write_joint_hierarchy(f, child_idx, skeleton, end_sites, indent + "  ")

    # Write end sites for this joint
    for parent_id, es_offset in end_sites:
        if parent_id == idx:
            ex, ey, ez = es_offset
            f.write(f"{indent}  End Site\n")
            f.write(f"{indent}  {{\n")
            f.write(f"{indent}    OFFSET {ex:.6f} {ey:.6f} {ez:.6f}\n")
            f.write(f"{indent}  }}\n")

    f.write(f"{indent}}}\n")


def remap_file(input_path, output_path, skeleton_type='cmu'):
    """Remap a single BVH file."""
    mapping = SKELETON_MAPPINGS.get(skeleton_type)
    if mapping is None:
        raise ValueError(f"Unknown skeleton type: {skeleton_type}. Options: {list(SKELETON_MAPPINGS.keys())}")

    positions, local_quats, names, parents, num_frames = extract_motion_data(input_path)
    mapped = build_ode_bvh(positions, local_quats, names, parents, mapping, output_path)

    return mapped, num_frames


def main():
    parser = argparse.ArgumentParser(description='Remap BVH skeleton to MoConVQ ODE format')
    parser.add_argument('--input', type=str, help='Input BVH file')
    parser.add_argument('--output', type=str, help='Output BVH file')
    parser.add_argument('--input-dir', type=str, help='Input directory of BVH files')
    parser.add_argument('--output-dir', type=str, help='Output directory for remapped BVH files')
    parser.add_argument('--skeleton', type=str, default='cmu',
                       choices=list(SKELETON_MAPPINGS.keys()),
                       help='Source skeleton type')
    args = parser.parse_args()

    if args.input and args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        mapped, frames = remap_file(args.input, args.output, args.skeleton)
        print(f"Remapped: {args.input} → {args.output}")
        print(f"  {mapped}/19 joints mapped, {frames} frames")

    elif args.input_dir and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        bvh_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.bvh')])
        print(f"Remapping {len(bvh_files)} BVH files from {args.input_dir}")
        print(f"  Skeleton type: {args.skeleton}")

        success = 0
        for i, bvh_file in enumerate(bvh_files):
            in_path = os.path.join(args.input_dir, bvh_file)
            out_path = os.path.join(args.output_dir, bvh_file)
            try:
                mapped, frames = remap_file(in_path, out_path, args.skeleton)
                success += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(bvh_files)}] {bvh_file}: {mapped}/19 joints, {frames}f")
            except Exception as e:
                print(f"  [{i+1}/{len(bvh_files)}] ✗ {bvh_file}: {e}")

        print(f"\nDone! {success}/{len(bvh_files)} files remapped.")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()