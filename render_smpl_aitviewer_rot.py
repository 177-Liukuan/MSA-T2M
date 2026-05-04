import argparse
import glob
import os

import numpy as np

from visualization.recover_visualize import recover_from_local_rotation


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
    """Optionally de-normalize 272-dim motion before rotation recovery.

    This keeps consistency with demo_t2m.py behavior: motion * Std + Mean.
    If your input is already de-normalized, pass --no_denorm.
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


def parse_272_to_smpl_inputs(motion_272, args):
    """Convert 272-dim motion into SMPLSequence inputs.

    Steps:
      1) optional de-normalization
      2) use repository inverse solver recover_from_local_rotation(..., 22)
      3) split returned SMPL85 into poses/trans/betas for aitviewer

    recover_from_local_rotation output layout: (F, 85)
      0:72   -> 24 joints axis-angle (flattened)
      72:78  -> reserved (zeros in this pipeline)
      78:81  -> root translation (x, y, z)
      81:85  -> beta tail (zeros in this pipeline)

    Note: For SMPLSequence, poses_root is first joint (3 dims),
    poses_body is remaining 23 joints (69 dims).
    """
    if motion_272.ndim != 2 or motion_272.shape[1] != 272:
        raise ValueError(f"Expected (F,272), got {motion_272.shape}")

    motion_for_recover = maybe_denorm_motion(motion_272, args)
    smpl_85 = recover_from_local_rotation(motion_for_recover, njoint=22).astype(np.float32)

    if smpl_85.ndim != 2 or smpl_85.shape[1] != 85:
        raise ValueError(f"Expected recovered SMPL85 shape (F,85), got {smpl_85.shape}")

    poses_72 = smpl_85[:, :72].astype(np.float32)
    trans = smpl_85[:, 78:81].astype(np.float32)

    poses_root = poses_72[:, :3]      # (F, 3)
    poses_body = poses_72[:, 3:72]    # (F, 69)

    # This pipeline uses fixed-zero shape by default; expose an override scalar if needed.
    if args.beta_value != 0.0:
        betas = np.full((10,), float(args.beta_value), dtype=np.float32)
    else:
        betas = np.zeros((10,), dtype=np.float32)

    return poses_root, poses_body, trans, betas


def build_smpl_sequence(motion_272, args, name):
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence

    poses_root, poses_body, trans, betas = parse_272_to_smpl_inputs(motion_272, args)

    # Point aitviewer to SMPL model root in this project.
    C.update_conf({"smplx_models": args.model_path, "export_dir": args.export_dir})

    smpl_layer = SMPLLayer(model_type="smpl", gender=args.gender, device=C.device)

    seq = SMPLSequence(
        poses_body=poses_body,
        poses_root=poses_root,
        trans=trans,
        betas=betas,
        smpl_layer=smpl_layer,
        color=tuple(args.color),
        name=name,
    )
    return seq


def render_single_file(args, motion_file):
    from aitviewer.headless import HeadlessRenderer
    from aitviewer.viewer import Viewer

    motion_272 = np.load(motion_file).astype(np.float32)
    if motion_272.ndim == 3 and motion_272.shape[0] == 1:
        motion_272 = motion_272[0]

    if motion_272.ndim != 2 or motion_272.shape[1] != 272:
        raise ValueError(f"Input npy must be (F,272) or (1,F,272), got {motion_272.shape}")

    name = os.path.basename(motion_file)
    smpl_seq = build_smpl_sequence(motion_272, args, name)

    if not args.save:
        viewer = Viewer(size=(args.width, args.height))
        viewer.run_animations = True
        viewer.playback_fps = args.fps
        viewer.scene.fps = args.fps
        viewer.scene.add(smpl_seq)
        viewer.center_view_on_node(smpl_seq)
        viewer.run()
        return

    os.makedirs(args.export_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(motion_file))[0]
    out_path = os.path.join(args.export_dir, f"{stem}.{args.format}")

    renderer = HeadlessRenderer(size=(args.width, args.height))
    renderer.run_animations = True
    renderer.playback_fps = args.fps
    renderer.scene.fps = args.fps
    renderer.scene.add(smpl_seq)
    renderer.lock_to_node(smpl_seq, (2, 2, 2), smooth_sigma=5.0)
    renderer.save_video(video_dir=out_path, output_fps=args.fps)
    print(f"[OK] Saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render 272-dim motions as SMPL mesh in aitviewer via rotation-based inverse recovery."
    )

    parser.add_argument("--motion_file", type=str, default=None, help="Single input .npy file with shape (F,272).")
    parser.add_argument("--motion_dir", type=str, default=None, help="Directory containing multiple (F,272) .npy files.")

    parser.add_argument("--mean_path", type=str, default="humanml3d_272/mean_std/Mean.npy")
    parser.add_argument("--std_path", type=str, default="humanml3d_272/mean_std/Std.npy")
    parser.add_argument(
        "--no_denorm",
        action="store_true",
        help="Disable motion * Std + Mean. Use this if input 272 is already de-normalized.",
    )

    parser.add_argument("--model_path", type=str, default="body_models/human_model_files")
    parser.add_argument("--gender", type=str, default="neutral", choices=["male", "female", "neutral"])
    parser.add_argument("--beta_value", type=float, default=0.0, help="Optional fixed value for all 10 betas.")

    parser.add_argument(
        "--color",
        type=float,
        nargs=4,
        default=[0.62, 0.73, 0.84, 1.0],
        help="Mesh RGBA color, e.g. 0.62 0.73 0.84 1.0",
    )

    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)

    parser.add_argument("--save", action="store_true", help="Enable headless export mode.")
    parser.add_argument("--export_dir", type=str, default="demo_output/aitviewer_renders_rot")
    parser.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])

    return parser.parse_args()


def main():
    args = parse_args()
    files = list_input_files(args)
    for motion_file in files:
        render_single_file(args, motion_file)


if __name__ == "__main__":
    main()
