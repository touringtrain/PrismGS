import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 构建数据
data = {
    'Method': ['TeTriRF', 'ReRF', 'Naive 3DGS', '4DGS', '3DGSstream', 'V^3', 'HiCoM'],
    'Category': ['NeRF-based', 'NeRF-based', 'GS-based', 'GS-based', 'GS-based', 'GS-based', 'GS-based'],
    'Size per frame (MB)': [0.15, 0.85, 18.3, 12.0, 7.8, 0.682, 0.9],
    'Rendering FPS': [8, 20, 200, 60, 215, 387, 274]
}
df = pd.DataFrame(data)

# 为不同类别和方法分配颜色和标记
categories = df['Category'].unique()
category_markers = {'NeRF-based': 'o', 'GS-based': '^'}
colors = plt.cm.tab10(np.linspace(0, 1, len(df['Method'])))

# 计算临界值曲线：在100Mbps下，满足1/30秒的每帧存储临界值（对数变换适配）
fps_range = np.linspace(5, 400, 100)
critical_size = []
for fps in fps_range:
    render_time = 1 / fps
    transmit_time = (1/30) - render_time
    if transmit_time <= 0:
        critical_size.append(0.01)  # 避免对数下为负，设置极小值
    else:
        critical_size.append(transmit_time * 100 * 10**6 / (8 * 1024 * 1024))

# 绘制图形
plt.figure(figsize=(10, 6))

# 首先填充右下角区域（在绘制其他元素之前）
# 创建填充区域的边界点
fill_x = np.concatenate([fps_range, [fps_range[-1], fps_range[0]]])
fill_y = np.concatenate([critical_size, [plt.ylim()[0], plt.ylim()[0]]])
plt.fill_between(fps_range, critical_size, plt.ylim()[0], 
                 color='lightgreen', alpha=0.3, label='Feasible Region\n(100Mbps, 30FPS)')

# 绘制临界曲线（对数坐标）
plt.plot(fps_range, critical_size, 'k--', linewidth=2, label='100Mbps Bandwidth Limit')

# 绘制散点（对数坐标）
for i, row in df.iterrows():
    marker = category_markers[row['Category']]
    plt.scatter(row['Rendering FPS'], row['Size per frame (MB)'], 
                color=colors[i], marker=marker, s=100, label=row['Method'])
    plt.text(row['Rendering FPS'] * 1.1, row['Size per frame (MB)'], 
             row['Method'], fontsize=12)  # 调整文本位置适配对数轴

# 设置对数坐标、标签和标题
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rendering FPS', fontsize=16)
plt.ylabel('Size per frame (MB)', fontsize=16)
plt.title('Dynamic 3D Rendering Methods: Rendering FPS vs Size per Frame', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlim(5, 400)
plt.ylim(0.1, 20)
plt.savefig('motivation_relatedwork.png', dpi=300, bbox_inches='tight')
