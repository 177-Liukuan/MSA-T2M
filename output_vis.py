import os
import numpy as np
import argparse
import matplotlib.pyplot as plt

# 配置
import skvideo
skvideo.setFFmpegPath(R"E:\ffmpeg\bin")

from aitviewer.viewer import Viewer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.configuration import CONFIG as C

# 你的旋转恢复函数
from visualization.recover_visualize import recover_from_local_rotation

# ====================== 配置 ======================
C.update_conf({
    "smplx_models": r"body_models/human_model_files",
})


# ====================== 你的数据解析函数 ======================
def maybe_denorm_motion(motion_272, args):
    if args.no_denorm:
        return motion_272.astype(np.float32)
    mean = np.load(args.mean_path).astype(np.float32)
    std = np.load(args.std_path).astype(np.float32)
    return (motion_272.astype(np.float32) * std + mean).astype(np.float32)


def parse_272_to_smpl(motion_file, args):
    motion_272 = np.load(motion_file).astype(np.float32)
    if motion_272.ndim == 3:
        motion_272 = motion_272[0]

    motion = maybe_denorm_motion(motion_272, args)
    smpl = recover_from_local_rotation(motion, njoint=22).astype(np.float32)

    poses = smpl[:, :72]
    trans = smpl[:, 72:75]

    poses_root = poses[:, :3]
    poses_body = poses[:, 3:]
    betas = np.zeros((10,), dtype=np.float32)

    return poses_root, poses_body, trans, betas


def infer_top_level_folder(input_path):
    """获取输入路径的第一层文件夹名。"""
    norm = os.path.normpath(str(input_path))
    _, tail = os.path.splitdrive(norm)
    tail = tail.lstrip("/\\")
    parts = [p for p in tail.split(os.sep) if p and p != "."]
    return parts[0] if parts else ""


def adapt_floor_to_trajectory(viewer, trans, floor_margin=1.25, floor_min_size=4.0):
    """基于轨迹主方向(PCA/SVD)自适应地板长宽、中心与旋转。

    设计目标：
    1) 在水平面提取主轴/次轴；
    2) 在主副轴上做投影并取跨度；
    3) 长宽分别自适应（不再强制正方形）；
    4) 地板中心与旋转和轨迹方向一致。
    """
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans shape should be (T,3), got {trans.shape}")

    if trans.shape[0] < 2:
        floor = viewer.scene.floor
        floor.scale = 1.0
        return

    # 选择水平面坐标与高度轴。
    if C.z_up:
        plane_axes = (0, 1)  # XY
        height_axis = 2
    else:
        plane_axes = (0, 2)  # XZ
        height_axis = 1

    a0, a1 = plane_axes
    pts2 = trans[:, [a0, a1]].astype(np.float32)

    # 1) SVD 提取主方向。
    mean2 = pts2.mean(axis=0)
    centered = pts2 - mean2
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    main = vh[0].astype(np.float32)
    main /= (np.linalg.norm(main) + 1e-8)

    # 次方向：与主方向垂直，并固定手性。
    if C.z_up:
        sec = np.array([-main[1], main[0]], dtype=np.float32)
    else:
        sec = np.array([main[1], -main[0]], dtype=np.float32)

    # 2) 投影后计算主轴/次轴跨度。
    p_main = centered @ main
    p_sec = centered @ sec
    min_main, max_main = float(p_main.min()), float(p_main.max())
    min_sec, max_sec = float(p_sec.min()), float(p_sec.max())

    length = max_main - min_main
    width = max_sec - min_sec

    # 3) 分别加 margin 并下限约束。
    length = max(length * float(floor_margin), float(floor_min_size))
    width = max(width * float(floor_margin), float(floor_min_size))

    # 4) 由投影中心恢复世界坐标中心。
    center2 = mean2 + main * ((min_main + max_main) * 0.5) + sec * ((min_sec + max_sec) * 0.5)

    floor = viewer.scene.floor

    # 4.1 位置：中心对齐 + 高度贴地。
    pos = floor.position.copy().astype(np.float32)
    pos[a0] = center2[0]
    pos[a1] = center2[1]
    pos[height_axis] = float(np.min(trans[:, height_axis])) - 1e-3
    floor.position = pos

    # 4.2 旋转：让地板边缘与主/次轴平行。
    if C.z_up:
        x_axis = np.array([main[0], main[1], 0.0], dtype=np.float32)
        y_axis = np.array([sec[0], sec[1], 0.0], dtype=np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)
    else:
        x_axis = np.array([main[0], 0.0, main[1]], dtype=np.float32)
        z_axis = np.array([sec[0], 0.0, sec[1]], dtype=np.float32)
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        rot = np.stack([x_axis, y_axis, z_axis], axis=1)

    floor.rotation = rot.astype(np.float32)

    # 4.3 将默认正方形网格改为矩形网格（length x width）。
    # 这样可以真正实现非正方形地板。
    floor.scale = 1.0
    if C.z_up:
        p0 = np.array([+length * 0.5, -width * 0.5, 0.0], dtype=np.float32)
        p1 = np.array([+length * 0.5, +width * 0.5, 0.0], dtype=np.float32)
        p2 = np.array([-length * 0.5, +width * 0.5, 0.0], dtype=np.float32)
        p3 = np.array([-length * 0.5, -width * 0.5, 0.0], dtype=np.float32)
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        p0 = np.array([+length * 0.5, 0.0, -width * 0.5], dtype=np.float32)
        p1 = np.array([+length * 0.5, 0.0, +width * 0.5], dtype=np.float32)
        p2 = np.array([-length * 0.5, 0.0, +width * 0.5], dtype=np.float32)
        p3 = np.array([-length * 0.5, 0.0, -width * 0.5], dtype=np.float32)
        normal = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    floor.vertices = np.row_stack([p1, p0, p2, p3]).astype(np.float32)
    floor.normals = np.tile(normal[None, :], (4, 1)).astype(np.float32)

    # 若 floor 已经创建了 GPU 缓冲，立即同步。
    if getattr(floor, "is_renderable", False):
        floor.vbo_vertices.write(floor.vertices.astype("f4").tobytes())
        floor.vbo_normals.write(floor.normals.astype("f4").tobytes())

# ====================== 轨迹图生成核心 ======================
def generate_motion_trail(args):
    print("加载运动数据...")
    poses_root, poses_body, trans, betas = parse_272_to_smpl(args.input, args)
    total_frames = len(poses_body)
    
    from matplotlib.colors import LinearSegmentedColormap

    cmap_A = LinearSegmentedColormap.from_list("modelA_blue", ["#E6F4FF", "#9DD3F9", "#3A8DDE", "#0B3C78"])
    cmap_B = LinearSegmentedColormap.from_list("modelB_orange", ["#FFF1E6", "#FFC78F", "#F07C2E", "#A84300"])
    cmap_C = LinearSegmentedColormap.from_list("modelC_green", ["#E8F8F1", "#A8E6C1", "#3FBF8F", "#0D6B4F"])

    top_folder = infer_top_level_folder(args.input)
    if top_folder == "MSA-T2M":
        cmap = cmap_A
        start_depth = 0.5
    elif top_folder == "MotionStreamer":
        cmap = cmap_B
        start_depth = 0.5  # 浅色B
    else:
        cmap = cmap_C
        start_depth = 0.5

    # 关键帧采样 + 蓝白渐变
    sample_frames = np.linspace(0, total_frames-1, args.num_samples, dtype=int)
    colors = [cmap(start_depth + (1 - start_depth) * (i / args.num_samples)) for i in range(args.num_samples)]


    v = Viewer(size=(1920, 1080))
    v.scene.floor.is_visible = True
    # 将棋盘地面改为统一浅灰色（关闭网格感）
    v.scene.floor.c1 = np.array((0.86, 0.86, 0.86, 1.0), dtype=np.float32)
    v.scene.floor.c2 = np.array((0.86, 0.86, 0.86, 1.0), dtype=np.float32)
    v.scene.floor.tiling = True
    v.scene.origin.enabled = False
    v.shadows_enabled = True
    v.auto_center = True
    v.scene.camera.smooth = False

    # 关键新增：按轨迹范围自适应地板大小
    adapt_floor_to_trajectory(
        v,
        trans,
        floor_margin=args.floor_margin,
        floor_min_size=args.floor_min_size,
    )
    
    
    # 预设视角
    v.scene.camera.position = np.array([0.0, 1.5, 4.0]) 
    v.scene.camera.target = np.array([0.0, 0.5, 0.0])

    smpl_layer = SMPLLayer(model_type="smpl", gender=args.gender, device=C.device)

    # 添加叠加人体
    for idx, f in enumerate(sample_frames):
        seq = SMPLSequence(
            poses_body=poses_body[f : f + 1],
            poses_root=poses_root[f : f + 1],
            trans=trans[f : f + 1],
            betas=betas,
            smpl_layer=smpl_layer,
            color=colors[idx],
            name=f"frame_{f}",
        )
        v.scene.add(seq)

    print("\n==================================================")
    print("  地板已根据轨迹范围自动缩放与居中")
    print("  请调整视角，然后按 P 键保存高清图片")
    print("  图片自动保存在当前目录下 screenshots 文件夹")
    print("==================================================\n")

    v.run()


# ====================== 主函数 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # 接受ACMMM
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MSA-T2M\MSA-T2M_a_person_runs_and_then_jumps_happily_because_the_acceptance_of_acmm_2026_20260401_212547.npy",
    # )
    
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MotionStreamer\MotionStreamer_a_person_runs_and_then_jumps_happily_because_the_acceptance_of_acmm_2026_20260401_213106.npy",
    # )
    
    ## 侧空翻
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MSA-T2M\MSA-T2M_a_person_performs_a_cartwheel_20260401_215957.npy",
    # )
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MotionStreamer\MotionStreamer_a_person_performs_a_cartwheel_20260401_220956.npy",
    # )
    
    ##躺下，站起，环形走路，躺回原地
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MSA-T2M\MSA-T2M_the_person_rises_from_a_laying_position_and_walks_in_a_clockwise_circle_and_then_lays_back_down_on_the_ground_20260401_141311.npy",
    # )
    parser.add_argument(
        "--input",
        type=str,
        default=r"MotionStreamer\MotionStreamer_the_person_rises_from_a_laying_position_and_walks_in_a_clockwise_circle_and_then_lays_back_down_on_the_ground_20260401_210032.npy",
    )
    
    
    ## 旋转跳舞
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MSA-T2M\MSA-T2M_a_person_is_dancing_swan_lake_20260401_231000.npy",
    # )
    
    ## 武术表表演
    # parser.add_argument(
    #     "--input",
    #     type=str,
    #     default=r"MSA-T2M\MSA-T2M_a_person_performing_martial_arts_20260402_003125.npy",
    # )
    
    parser.add_argument("--mean_path", type=str, default="Mean.npy")
    parser.add_argument("--std_path", type=str, default="Std.npy")
    parser.add_argument("--no_denorm", action="store_true")
    parser.add_argument("--gender", type=str, default="neutral")
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--output", type=str, default="motion_trail.png")

    # 地板自适应参数
    parser.add_argument("--floor_margin", type=float, default=1.5, help="地板相对轨迹的留白比例")
    parser.add_argument("--floor_min_size", type=float, default=1, help="地板最小边长")

    args = parser.parse_args()

    generate_motion_trail(args)
