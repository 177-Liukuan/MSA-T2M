"""
HumanML3D-272 数据集可视化增强版 (符合 XZ-Ground, Y-Up 规范)
改进点：
1. 路径修复：确保所有结果保存在脚本所在目录的 vis_res 下。
2. 坐标映射：将数据 Y 映射到绘图 Z，实现 XZ 为地面。
3. 视角优化：将地面置于底部，设置灰色半透明地面。
4. 动态联动：修改 SAMPLE_ID，所有内容同步更新。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sys
import os

# ================= 配置区 =================
SAMPLE_ID = "000382"

DATA_ROOT = 'humanml3d_272'
MOTION_DIR = os.path.join(DATA_ROOT, 'motion_data')
TEXT_DIR = os.path.join(DATA_ROOT, 'texts')

# 动态获取当前脚本所在的绝对路径，并创建输出文件夹
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'vis_res')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================

# 22关节骨架连接关系 (SMPL 标准)
SKELETON_CHAIN = [
    [0, 3, 6, 9, 12, 15],  # 脊柱 -> 头部
    [0, 1, 4, 7, 10],      # 左腿
    [0, 2, 5, 8, 11],      # 右腿
    [9, 13, 16, 18, 20],   # 左臂
    [9, 14, 17, 19, 21],   # 右臂
]


def load_data(sample_id):
    """动态加载数据和文本"""
    npy_path = os.path.join(MOTION_DIR, f"{sample_id}.npy")
    txt_path = os.path.join(TEXT_DIR, f"{sample_id}.txt")

    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"找不到数据文件: {npy_path}")

    motion_272 = np.load(npy_path)

    description = "Unknown Motion"
    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            description = f.readline().split('#')[0]

    return motion_272, description


def recover_positions(motion_272):
    """恢复世界坐标位移"""
    frames = motion_272.shape[0]
    # 8:74 是局部坐标 (22 joints * 3)
    joints_local = motion_272[:, 8:74].reshape(frames, 22, 3)

    vx, vz = motion_272[:, 0], motion_272[:, 1]
    curr_x, curr_z = 0.0, 0.0
    trajectory = np.zeros((frames, 3))
    for i in range(1, frames):
        curr_x += vx[i-1]
        curr_z += vz[i-1]
        trajectory[i, 0] = curr_x
        trajectory[i, 2] = curr_z

    joints_world = joints_local.copy()
    for j in range(22):
        joints_world[:, j, :] += trajectory

    return joints_world


def setup_ground(ax, joints, color='gray'):
    """设置灰色地面并将其置于底部"""
    x_min, x_max = joints[:, :, 0].min(), joints[:, :, 0].max()
    z_min, z_max = joints[:, :, 2].min(), joints[:, :, 2].max()
    padding = 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min-padding, x_max+padding, 10),
        np.linspace(z_min-padding, z_max+padding, 10)
    )
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color=color, zorder=-1)

    ax.set_zlim(0, max(1.5, joints[:, :, 1].max() + 0.2))
    ax.set_xlim(x_min-padding, x_max+padding)
    ax.set_ylim(z_min-padding, z_max+padding)


def visualize_static(joints, sample_id, description):
    """绘制静态姿态序列"""
    num_poses = 5
    indices = np.linspace(0, len(joints)-1, num_poses, dtype=int)
    fig = plt.figure(figsize=(15, 4))
    fig.suptitle(f"ID: {sample_id} | {description}", fontsize=12, y=0.95)

    for i, idx in enumerate(indices):
        ax = fig.add_subplot(1, num_poses, i+1, projection='3d')
        j_frame = joints[idx]
        for chain in SKELETON_CHAIN:
            ax.plot(j_frame[chain, 0], j_frame[chain, 2],
                    j_frame[chain, 1], 'b-o', markersize=2)

        setup_ground(ax, joints)
        ax.view_init(elev=15, azim=45)
        ax.set_axis_off()
        ax.set_title(f"Frame {idx}", fontsize=8)

    save_path = os.path.join(OUTPUT_DIR, f"static_{sample_id}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()  # 释放内存
    print(f"✅ 静态预览已保存: {save_path}")


def create_gif(joints, sample_id, description, fps=20):
    """生成 3D 骨架动画"""
    save_path = os.path.join(OUTPUT_DIR, f"anim_{sample_id}.gif")
    print(f"🎬 正在生成 ID {sample_id} 的动画...")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.clear()
        j_frame = joints[frame]
        for chain in SKELETON_CHAIN:
            ax.plot(j_frame[chain, 0], j_frame[chain, 2], j_frame[chain, 1],
                    color='red', linewidth=3, marker='o', markerfacecolor='black', markersize=4)
        setup_ground(ax, joints)
        ax.set_xlabel('X (World)')
        ax.set_ylabel('Z (World)')
        ax.set_zlabel('Height (Y)')
        ax.set_title(f"ID: {sample_id}\n{description[:50]}...")
        ax.view_init(elev=20, azim=45 + frame/2)

    anim = FuncAnimation(fig, update, frames=len(joints), interval=1000/fps)
    anim.save(save_path, writer='pillow', fps=fps)
    plt.close()
    print(f"✅ 动画已保存: {save_path}")


def main():
    try:
        motion_272, description = load_data(SAMPLE_ID)
        print(f"📦 加载完成: {SAMPLE_ID}")
        print(f"📝 描述: {description}")

        joints_world = recover_positions(motion_272)
        visualize_static(joints_world, SAMPLE_ID, description)

        choice = input("\n是否生成动画? [y/N]: ")
        if choice.lower() == 'y':
            create_gif(joints_world, SAMPLE_ID, description)

    except Exception as e:
        print(f"❌ 出错: {e}")


if __name__ == '__main__':
    main()
