import matplotlib.pyplot as plt
import numpy as np


import matplotlib.font_manager as fm
font_path = './TIMES.TTF'
fm.fontManager.addfont(font_path)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'


# 数据
qp_values = [0, 10, 20, 30, 40, 50]
# attributes = ['position_low', 'color', 'opacity', 'scale', 'rotation']
attributes = ['color', 'opacity', 'scale']
psnr_data = {
    # 'position_low': [36.2693, 36.2689, 36.2670, 36.2526, 36.1352, 35.1087],
    'color': [36.2693, 36.2084, 35.6170, 31.6809, 23.8974, 21.7541],
    'opacity': [36.2693, 36.2691, 36.2577, 36.2007, 35.1503, 27.4803],
    'scale': [36.2693, 36.2552, 36.0955, 33.2235, 19.2954, 19.1380],
    # 'rotation': [36.2693, 36.1731, 35.4673, 30.9327, 22.6686, 22.1854]
}

for attr in attributes:
    psnr_data[attr] = [val / 36.2693 for val in psnr_data[attr]]

# 创建图表
plt.figure(figsize=(7, 6))

# 定义颜色和标记
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
colors = ['#0C5DA5', '#FF2C00', '#00B945']
markers = ['o', 's', '^', 'D', 'v']
linestyles = ['-', '--', '-.', '--', '-']

# 绘制每条线
for i, attr in enumerate(attributes):
    plt.plot(qp_values, psnr_data[attr], 
             label=attr, 
             color=colors[i], 
             marker=markers[i],
             linestyle=linestyles[i],
             markersize=12,
             linewidth=3,
             markeredgecolor='white',
             markeredgewidth=1)

# 设置图表属性
plt.xlabel('Quantization Step', fontsize=22)
plt.ylabel('Relative Visual Quality', fontsize=22)
# plt.title('Impact of Gaussian Splatting Attributes on PSNR at Different QP Values', 
        #   fontsize=16, fontweight='bold', pad=20)

# 设置坐标轴范围
plt.xlim(-2, 52)
plt.ylim(0.5, 1.05)

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 添加图例
plt.legend(fontsize=22, framealpha=0.9, loc='lower left')

# 设置刻度标签
plt.xticks(qp_values, fontsize=22)
plt.yticks(fontsize=22)

# 调整布局
plt.tight_layout()

# 显示图表
# plt.show()

# 可选：保存图表
plt.savefig('motivation_fi_analysis_truncated.pdf', dpi=300, bbox_inches='tight')