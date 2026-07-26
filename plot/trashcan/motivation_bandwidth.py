import matplotlib.pyplot as plt
import numpy as np

# 设置科研论文风格
plt.rcParams.update({
    'figure.dpi': 300,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'lines.linewidth': 1.2,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05
})

# 数据准备
methods = ['Naive 3DGS', '4DGS', '3DGStream', 'V$^3$', 'HiCoM']
bandwidth = [4392, 2880, 1872, 163.68, 216]

# 创建画布和子图
fig, ax = plt.subplots(figsize=(6.5, 4))

# 使用科研论文风格的浅色系
colors = ['#8DA0CB', '#FC8D62', '#66C2A5', '#E78AC3', '#A6D854']

# 绘制柱状图
bars = ax.bar(methods, bandwidth, color=colors, 
              edgecolor='black', linewidth=0.6, alpha=0.8, width=0.7)

# 设置纵坐标为对数刻度
ax.set_yscale('log')
ax.set_ylabel('Bandwidth (Mbps)', fontweight='bold')
ax.set_xlabel('Methods', fontweight='bold')

# 设置标题（科研论文通常标题简洁）
ax.set_title('Bandwidth Comparison of GS-based Methods', fontweight='bold', pad=15)

# 为每个柱子添加数值标签
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height * 1.08,
            f'{height:.1f}' if height < 1000 else f'{height:.0f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# 绘制参考线
ax.axhline(y=80, color='#2E8B57', linestyle='--', linewidth=1.2, 
           alpha=0.8, label='80 Mbps (Reference)')
ax.axhline(y=210, color='#FF8C00', linestyle='--', linewidth=1.2, 
           alpha=0.8, label='210 Mbps (Reference)')

# 设置坐标轴样式
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.8)
ax.spines['bottom'].set_linewidth(0.8)

# 添加网格线（科研论文常用浅色网格）
ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)

# 设置y轴刻度
ax.set_ylim(10, 10000)
yticks = [10, 100, 1000, 10000]
ax.set_yticks(yticks)
ax.set_yticklabels([f'{x}' for x in yticks])

# 旋转x轴标签以避免重叠
# plt.xticks(rotation=15, ha='right')

# 显示图例
ax.legend(frameon=True, fancybox=False, edgecolor='black', 
          facecolor='white', framealpha=0.8)

# 调整布局
plt.tight_layout()

# 显示图形
plt.savefig('motivation_bandwidth.png', dpi=300)
plt.savefig('motivation_bandwidth.pdf')