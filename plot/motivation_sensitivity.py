import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 准备各属性的DataFrame
df_position = pd.DataFrame({
    'qp': [0, 10, 20, 30, 40, 50],
    'x': [34.7459, 34.7459, 34.7456, 34.7436, 34.7338, 34.6373],
    'y': [34.7461, 34.7460, 34.7456, 34.7383, 34.6732, 34.0998],
    'z': [34.7460, 34.7458, 34.7456, 34.7437, 34.7234, 34.5758]
})

df_color = pd.DataFrame({
    'qp': [0, 10, 20, 30, 40, 50],
    'R': [34.8975, 34.8843, 34.7456, 33.6180, 31.0272, 30.4834],
    'G': [34.9475, 34.9241, 34.7456, 33.5104, 30.8131, 30.2863],
    'B': [34.8838, 34.8759, 34.7456, 33.7072, 31.4938, 30.4642]
})

df_scale = pd.DataFrame({
    'qp': [0, 10, 20, 30, 40, 50],
    'x': [34.8060, 34.8035, 34.7456, 34.0495, 24.2244, 31.7831],
    'y': [34.8137, 34.8096, 34.7456, 33.6797, 22.0563, 25.6026],
    'z': [34.8464, 34.8034, 34.7456, 33.9829, 24.5808, 31.0262]
})

df_rotation = pd.DataFrame({
    'qp': [0, 10, 20, 30, 40, 50],
    'x': [34.8262, 34.8117, 34.7456, 33.9307, 29.5515, 27.4475],
    'y': [35.0034, 34.9621, 34.7456, 32.7834, 26.5781, 25.5664],
    'z': [34.9013, 34.8745, 34.7456, 33.5619, 28.6248, 24.7880],
    'w': [34.9519, 34.9282, 34.7456, 33.2807, 27.4044, 25.9310]
})

def plot_attribute(ax, df, title, ylabel, columns):
    for col in columns:
        ax.plot(df['qp'], df[col], marker='o', linewidth=4, markersize=10, label=col)
    ax.set_title(title, fontsize=22, pad=15)  # 增加标题与图之间的间距
    ax.set_xlabel('QP Value', fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks(df['qp'])
    ax.tick_params(axis='both', labelsize=18)  # 增加刻度标签大小
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(fontsize=18, loc='best')  # 增加图例字体大小

# 创建更大的画布并调整子图间距
fig, axes = plt.subplots(2, 2, figsize=(20, 14))

# 绘制四个子图
plot_attribute(axes[0, 0], df_position, 'Position (x, y, z) vs QP', 'Value', ['x', 'y', 'z'])
plot_attribute(axes[0, 1], df_color, 'Color (R, G, B) vs QP', 'Value', ['R', 'G', 'B'])
plot_attribute(axes[1, 0], df_scale, 'Scale (x, y, z) vs QP', 'Value', ['x', 'y', 'z'])
plot_attribute(axes[1, 1], df_rotation, 'Rotation (x, y, z, w) vs QP', 'Value', ['x', 'y', 'z', 'w'])

# 调整整体布局，为主标题留出更多空间
plt.suptitle('Attribute Values vs Compression QP', fontsize=22, y=0.95)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle保留空间
plt.savefig('motivation_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig('motivation_sensitivity.pdf', bbox_inches='tight')
