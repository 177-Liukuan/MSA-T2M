import argparse
import glob
import os

import numpy as np

from visualization.recover_visualize import recover_from_local_position


def list_input_files(args):
    if args.motion_file is not None:
        return [args.motion_file]

    if args.motion_dir is not None:
        files = sorted(glob.glob(os.path.join(args.motion_dir, "*.npy")))
        if len(files) == 0:
            raise RuntimeError(f"No .npy file found in directory: {args.motion_dir}")
        return files

    raise ValueError("Please provide --motion_file or --motion_dir.")


def maybe_denorm_motion(motion_272, args):
    """Optionally de-normalize decoded motion, matching demo_t2m.py behavior.

    demo_t2m.py uses: motion * Std + Mean before recover_from_local_position.
    This function applies the same rule when --no_denorm is NOT set.
    """
    if args.no_denorm:
        return motion_272.astype(np.float32)

    if not os.path.exists(args.mean_path) or not os.path.exists(args.std_path):
        raise FileNotFoundError(
            f"mean/std file not found: mean={args.mean_path}, std={args.std_path}. "
            "Use --no_denorm if your input is already de-normalized."
        )

    mean = np.load(args.mean_path).astype(np.float32)
    std = np.load(args.std_path).astype(np.float32)
    if mean.shape[0] != 272 or std.shape[0] != 272:
        raise ValueError(f"Mean/Std must have shape (272,), got mean={mean.shape}, std={std.shape}")

    return (motion_272.astype(np.float32) * std + mean).astype(np.float32)


def parse_272_to_joint_positions(motion_272, args):
    """Convert 272-dim motion into global joint positions (F,22,3).

    We deliberately follow demo_t2m.py style:
      1) optional de-normalization with Mean/Std
      2) recover_from_local_position(..., 22)

    The recover function internally handles the 272 layout:
      - 0:2      root xz velocity
      - 2:8      root heading 6D
      - 8:74     local joint positions (22*3)
      - ...      other blocks
    and reconstructs global coordinates over time.
    """
    if motion_272.ndim != 2 or motion_272.shape[1] != 272:
        raise ValueError(f"Expected (F,272), got {motion_272.shape}")

    motion_for_recover = maybe_denorm_motion(motion_272, args)
    xyz = recover_from_local_position(motion_for_recover, 22).astype(np.float32)
    if xyz.ndim != 3 or xyz.shape[1:] != (22, 3):
        raise ValueError(f"Recovered joint positions should be (F,22,3), got {xyz.shape}")
    return xyz


def build_joint_connections_22():
    """Build 22-joint SMPL-core kinematic edges for Skeletons renderable."""
    parents = np.array([
        -1,  # 0 pelvis
        0, 0, 0,  # 1,2,3
        1, 2, 3,  # 4,5,6
        4, 5, 6,  # 7,8,9
        7, 8, 9,  # 10,11,12
        9, 9, 12,  # 13,14,15
        13, 14, 16, 17, 18, 19  # 16..21
    ], dtype=np.int32)

    edges = []
    for joint_idx in range(22):
        p = int(parents[joint_idx])
        if p >= 0:
            edges.append([p, joint_idx])
    return np.asarray(edges, dtype=np.int32)


def build_skeleton_renderable(joint_positions, args, name):
    from aitviewer.renderables.skeletons import Skeletons

    connections = build_joint_connections_22()
    return Skeletons(
        joint_positions=joint_positions,
        joint_connections=connections,
        radius=args.joint_radius,
        color=tuple(args.color),
        name=name,
    )


def render_single_file(args, motion_file):
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.viewer import Viewer

    motion_272 = np.load(motion_file).astype(np.float32)
    if motion_272.ndim == 3 and motion_272.shape[0] == 1:
        motion_272 = motion_272[0]

    joint_positions = parse_272_to_joint_positions(motion_272, args)
    name = os.path.basename(motion_file)
    skeleton = build_skeleton_renderable(joint_positions, args, name)

    if not args.save:
        viewer = Viewer(size=(args.width, args.height))
        viewer.run_animations = True
        viewer.playback_fps = args.fps
        viewer.scene.fps = args.fps
        viewer.scene.add(skeleton)
        viewer.center_view_on_node(skeleton)
        viewer.run()
        return

    os.makedirs(args.export_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(motion_file))[0]
    out_path = os.path.join(args.export_dir, f"{stem}.{args.format}")

    renderer = HeadlessRenderer(size=(args.width, args.height))
    renderer.run_animations = True
    renderer.playback_fps = args.fps
    renderer.scene.fps = args.fps
    renderer.scene.add(skeleton)
    renderer.lock_to_node(skeleton, (2, 2, 2), smooth_sigma=5.0)
    renderer.save_video(video_dir=out_path, output_fps=args.fps)
    print(f"[OK] Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 272-dim motions in aitviewer using recovered joint positions (Skeletons)."
    )

    parser.add_argument("--motion_file", type=str, default=R"demo_output\generated_272\a_person_walks_forward_turns_then_sits_then_stands_and_walks_back_20260322_130237_961_272.npy", help="Single input .npy file with shape (F,272).")
    parser.add_argument("--motion_dir", type=str, default=None, help="Directory containing multiple (F,272) .npy files.")

    parser.add_argument("--mean_path", type=str, default="Mean.npy")
    parser.add_argument("--std_path", type=str, default="Std.npy")
    parser.add_argument(
        "--no_denorm",
        action="store_true",
        help="Disable motion * Std + Mean. Use this if input 272 is already de-normalized.",
    )

    parser.add_argument(
        "--color",
        type=float,
        nargs=4,
        default=[0.62, 0.73, 0.84, 1.0],
        help="Skeleton RGBA color, e.g. 0.62 0.73 0.84 1.0",
    )
    parser.add_argument("--joint_radius", type=float, default=0.02, help="Joint sphere radius for skeleton rendering.")

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument("--save", action="store_true", help="Enable headless export mode.")
    parser.add_argument("--export_dir", type=str, default="demo_output/aitviewer_renders")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])

    return parser.parse_args()


def main():
    args = parse_args()
    files = list_input_files(args)
    for motion_file in files:
        render_single_file(args, motion_file)


if __name__ == "__main__":
    main()
