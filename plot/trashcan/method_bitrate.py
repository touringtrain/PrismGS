import matplotlib.pyplot as plt
import numpy as np

# 数据
qp_levels = np.array([0, 10, 20, 30, 40, 50])
bitrate_data = {
    'Position': np.array([504, 420, 288, 192, 114, 24]),
    'Color': np.array([108.0, 90.0, 42.0, 24.0, 9.0, 2.4]),
    'Opacity': np.array([54.0, 42.0, 19.0, 8.0, 2.4, 0.8]),
    'Scale': np.array([177.0, 144.0, 63.0, 34.5, 21.0, 2.4]),
    'Rotation': np.array([220.0, 152.0, 72.0, 36.0, 16.0, 3.2])
}

# 计算总码率和比例
total_bitrate = np.sum([bitrate_data[attr] for attr in bitrate_data], axis=0)
percentages = {}
for attr in bitrate_data:
    percentages[attr] = (bitrate_data[attr] / total_bitrate) * 100

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 左图：总体积变化（柱状图）
colors_total = ['#1f77b4']
bars = ax1.bar(qp_levels, total_bitrate, width=4, 
               color=colors_total[0], alpha=0.7, edgecolor='black')

# 在柱子上添加数值标签
for bar, total in zip(bars, total_bitrate):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 10,
             f'{total:.0f} KB', ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

ax1.set_xlabel('Quantization Parameter (QP)', fontsize=16)
ax1.set_ylabel('Total Bitrate (KB)', fontsize=16)
ax1.set_title('Total Storage Size vs QP', fontsize=18, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_xticks(qp_levels)
ax1.set_ylim(0, max(total_bitrate) * 1.15)

# 右图：各属性比例（堆叠柱状图）
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
attributes = list(bitrate_data.keys())
bottom = np.zeros(len(qp_levels))

for i, attr in enumerate(attributes):
    bars = ax2.bar(qp_levels, percentages[attr], bottom=bottom, 
                   label=attr, color=colors[i], alpha=0.8, width=4)
    bottom += percentages[attr]

ax2.set_xlabel('Quantization Parameter (QP)', fontsize=16)
ax2.set_ylabel('Percentage (%)', fontsize=16)
ax2.set_title('Bitrate Distribution by Attribute', fontsize=18, fontweight='bold')
ax2.legend(fontsize=12, framealpha=0.9)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(qp_levels)
ax2.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('method_bitrate_stacked.pdf', dpi=300, bbox_inches='tight')
plt.savefig('method_bitrate_stacked.png', dpi=300, bbox_inches='tight')
