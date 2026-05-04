import warnings
from textwrap import wrap
import imageio
import sys
import os
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 关闭警告信息
warnings.filterwarnings('ignore')

# ==================== 输出目录管理 ====================


def get_output_dir():
    """获取vis_res输出目录,如果不存在则创建"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, 'vis_res')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# ==================== 骨架连接定义 ====================

SKELETON_CHAIN_22 = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]

SKELETON_CHAIN_21 = [
    [0, 11, 12, 13, 14, 15],
    [0, 16, 17, 18, 19, 20],
    [0, 1, 2, 3, 4],
    [3, 5, 6, 7],
    [3, 8, 9, 10]
]

COLORS = ['red', 'blue', 'black', 'green', 'orange',
          'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
          'darkred', 'darkred', 'darkred', 'darkred', 'darkred']


# ==================== 数据加载和提取 ====================

def load_text_label(npy_path):
    """
    根据npy路径自动寻找对应的txt标签文件并读取内容
    """
    try:
        # 路径替换：将 motion_data 替换为 texts, 将 .npy 替换为 .txt
        txt_path = npy_path.replace(
            'motion_data', 'texts').replace('.npy', '.txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().split('#')[0]  # 只取第一个#之前的原始文本
                print(f"📖 提取到文本标签: {first_line}")
                return first_line
        else:
            print(f"⚠️ 未找到标签文件: {txt_path}")
            return "No Label Found"
    except Exception as e:
        print(f"❌ 读取标签失败: {e}")
        return "Error Loading Label"


def load_motion_data(data_path):
    motion = np.load(data_path)
    print(f"✅ 加载运动数据: {data_path}")
    print(f"   Shape: {motion.shape}")
    return motion


def extract_joints_positions(motion_272):
    num_frames = motion_272.shape[0]
    num_joints = 22
    # 提取局部位置 (维度 8:74)
    local_positions = motion_272[:, 8:8+3 *
                                 num_joints].reshape(num_frames, num_joints, 3)
    return local_positions


# ==================== 3D可视化核心函数 ====================

def plot_3d_motion_frame(joints, frame_idx, title=None, figsize=(10, 10), azim=-90):
    num_joints = joints.shape[1]
    skeleton_chain = SKELETON_CHAIN_22 if num_joints == 22 else SKELETON_CHAIN_21

    data = joints.copy()
    MINS = data.min(axis=0).min(axis=0)

    # Y轴高度归零
    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    # 将关节位置相对于根关节归零
    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    fig = plt.figure(figsize=figsize, dpi=96)
    if title is not None:
        wrapped_title = '\n'.join(wrap(title, 40))
        fig.suptitle(wrapped_title, fontsize=16)

    ax = p3.Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    limits = 2
    ax.set_xlim(-limits, limits)
    ax.set_ylim(-limits, limits)
    ax.set_zlim(0, limits)
    ax.grid(False)

    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    if frame_idx > 1:
        ax.plot3D(trajec[:frame_idx, 0] - trajec[frame_idx, 0],
                  np.zeros_like(trajec[:frame_idx, 0]),
                  trajec[:frame_idx, 1] - trajec[frame_idx, 1],
                  linewidth=1.5, color='blue', alpha=0.6)

    frame_data = data[frame_idx]
    for i, (chain, color) in enumerate(zip(skeleton_chain, COLORS)):
        linewidth = 4.0 if i < 5 else 2.0
        ax.plot3D(frame_data[chain, 0],
                  frame_data[chain, 2],
                  frame_data[chain, 1],
                  linewidth=linewidth, color=color)

    ax.view_init(elev=20, azim=azim)
    ax.dist = 7.5
    return fig, ax


def create_static_poses(joints, save_name='motion', num_poses=6):
    num_frames = len(joints)
    frame_indices = np.linspace(0, num_frames-1, num_poses, dtype=int)
    fig = plt.figure(figsize=(20, 4), dpi=150)
    fig.suptitle(f'{save_name} - Poses Comparison',
                 fontsize=16, fontweight='bold')

    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(1, num_poses, i+1, projection='3d')
        num_joints = joints.shape[1]
        skeleton_chain = SKELETON_CHAIN_22 if num_joints == 22 else SKELETON_CHAIN_21
        joints_frame = joints[frame_idx]

        ax.scatter(joints_frame[:, 0], joints_frame[:, 2],
                   joints_frame[:, 1], c='red', s=20, alpha=0.8)
        for chain in skeleton_chain:
            for j in range(len(chain)-1):
                start, end = joints_frame[chain[j]], joints_frame[chain[j+1]]
                ax.plot([start[0], end[0]], [start[2], end[2]], [
                        start[1], end[1]], 'b-', linewidth=2, alpha=0.7)

        ax.set_title(f'Frame {frame_idx}')
        ax.view_init(elev=20, azim=45)
        ax.grid(False)

    plt.tight_layout()
    save_path = os.path.join(get_output_dir(), f'{save_name}_poses.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 静态姿势图已保存: {save_path}")


def create_animation(joints, save_name='motion', fps=30, title=None):
    print(f"\n🎬 正在创建动画 (FPS={fps})...")
    num_frames = joints.shape[0]
    frames = []
    for frame_idx in range(0, num_frames, 2):  # 每隔2帧采样
        fig, ax = plot_3d_motion_frame(joints, frame_idx, title=title)
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
        if (frame_idx + 1) % 20 == 0:
            print(f"   处理进度: {frame_idx+1}/{num_frames}")

    save_path = os.path.join(get_output_dir(), f'{save_name}_animation.gif')
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"✅ 动画已保存: {save_path}")


def create_trajectory_plot(joints, save_name='motion'):
    root_trajectory = joints[:, 0, :]
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f'{save_name} - Trajectory Analysis',
                 fontsize=14, fontweight='bold')

    ax1 = fig.add_subplot(131)
    ax1.plot(root_trajectory[:, 0], root_trajectory[:, 2], 'b-', alpha=0.6)
    ax1.set_title('XZ Plane (Top View)')

    ax2 = fig.add_subplot(132)
    ax2.plot(root_trajectory[:, 0], root_trajectory[:, 1], 'g-', alpha=0.6)
    ax2.set_title('XY Plane (Side View)')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot(root_trajectory[:, 0], root_trajectory[:,
             2], root_trajectory[:, 1], 'b-', alpha=0.6)
    ax3.set_title('3D Trajectory')

    plt.tight_layout()
    save_path = os.path.join(get_output_dir(), f'{save_name}_trajectory.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 轨迹图已保存: {save_path}")


# ==================== 主函数 (在此修改变量) ====================

def main():
    # ------------------ 配置区域 ------------------
    # 在这里直接修改你想要可视化的文件路径和参数
    MOTION_FILE = 'humanml3d_272/motion_data/000526.npy'
    SAVE_NAME = 'sample_000008_with_text'
    FPS = 30
    NUM_POSES = 6
    GENERATE_ANIMATION = True
    # ---------------------------------------------

    print("="*60)
    print("运动数据可视化工具 (代码内置配置模式)")
    print("="*60)

    # 1. 自动提取文本标签
    text_label = load_text_label(MOTION_FILE)

    # 2. 加载运动数据
    motion_272 = load_motion_data(MOTION_FILE)

    # 3. 提取关节位置
    joints = extract_joints_positions(motion_272)

    # 4. 生成可视化
    create_static_poses(joints, save_name=SAVE_NAME, num_poses=NUM_POSES)
    create_trajectory_plot(joints, save_name=SAVE_NAME)

    if GENERATE_ANIMATION:
        # 将提取到的文本标签作为标题传入
        create_animation(joints, save_name=SAVE_NAME,
                         fps=FPS, title=text_label)

    print("\n" + "="*60)
    print(f"📁 所有可视化完成! 输出目录: {get_output_dir()}")
    print("="*60)


if __name__ == '__main__':
    main()
