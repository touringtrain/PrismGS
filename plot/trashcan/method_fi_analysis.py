import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图片的文件夹
output_dir = "attribute_quality_plots"
os.makedirs(output_dir, exist_ok=True)

# 原始实验数据
qp_levels = np.array([0, 10, 20, 30, 40, 50])
quality_data_1 = {
    'Position': np.array([34.7470, 34.7466, 34.7456, 34.7352, 34.6413, 33.8800]),
    'Color': np.array([35.2377, 35.1931, 34.7456, 31.3446, 23.8429, 21.7427]),
    'Opacity': np.array([34.7560, 34.7555, 34.7456, 34.7009, 33.9201, 27.2833]),
    'Scale': np.array([34.9311, 34.9214, 34.7456, 32.3086, 19.3342, 19.3121]),
    'Rotation': np.array([35.4619, 35.3728, 34.7456, 30.6098, 22.6330, 23.1701])
}

# 新实验数据
quality_data_2 = {
    'Position_low': np.array([33.8226, 33.8225, 33.8224, 33.8154, 33.7604, 33.2019]),
    'Color': np.array([34.1960, 34.1441, 33.8224, 31.2463, 24.8799, 22.8114]),
    'Opacity': np.array([33.8243, 33.8231, 33.8224, 33.7775, 33.1695, 27.7290]),
    'Scale': np.array([33.9029, 33.8900, 33.8224, 32.4383, 22.3361, 21.3760]),
    'Rotation': np.array([34.1902, 34.1422, 33.8224, 31.5641, 25.0860, 24.0696])
}

# 计算归一化质量函数 f_i(qp) - 两组数据分别计算
normalized_quality_1 = {}
for attr, qualities in quality_data_1.items():
    base_quality = qualities[2]  # QP=20时的质量
    normalized_quality_1[attr] = qualities / base_quality

normalized_quality_2 = {}
for attr, qualities in quality_data_2.items():
    # 注意：第二组数据中Position改名为Position_low
    if attr == 'Position_low':
        base_quality = qualities[2]  # QP=20时的质量
        normalized_quality_2['Position'] = qualities / base_quality
    else:
        base_quality = qualities[2]  # QP=20时的质量
        normalized_quality_2[attr] = qualities / base_quality

# 计算平均值和标准差
average_quality = {}
std_quality = {}

attributes = ['Position', 'Color', 'Opacity', 'Scale', 'Rotation']
for attr in attributes:
    # 获取两组数据
    data1 = normalized_quality_1[attr]
    data2 = normalized_quality_2[attr]
    
    # 计算平均值和标准差
    average_quality[attr] = np.mean([data1, data2], axis=0)
    std_quality[attr] = np.std([data1, data2], axis=0, ddof=1)  # 样本标准差

# 颜色和标记样式
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
markers = ['o', 's', '^', 'D', 'v']
line_styles = ['-', '--', '-.', ':', '-']

# 为每个属性单独创建图表
for i, attr in enumerate(attributes):
    # 创建单个属性的图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.05)  # 减少两个子图之间的间距
    
    avg_data = average_quality[attr]
    std_data = std_quality[attr]
    
    # 上图：显示QP=0,10,20的细节
    ax1.plot(qp_levels[:3], avg_data[:3], 
             color=colors[i], marker=markers[i], markersize=10,
             linewidth=3, label=attr, linestyle=line_styles[i])
    
    # 添加误差线
    # ax1.errorbar(qp_levels[:3], avg_data[:3], yerr=std_data[:3],
    #              color=colors[i], capsize=7, capthick=3, elinewidth=3,
    #              alpha=0.7, linestyle='')
    
    # 下图：显示完整范围
    ax2.plot(qp_levels, avg_data, 
             color=colors[i], marker=markers[i], markersize=10,
             linewidth=3, label=attr, linestyle=line_styles[i])
    
    # 添加误差线
    # ax2.errorbar(qp_levels, avg_data, yerr=std_data,
    #              color=colors[i], capsize=5, capthick=2, elinewidth=2,
    #              alpha=0.7, linestyle='')
    
    # 设置上图（细节部分：QP=0,10,20）
    print(avg_data)
    ax1.set_ylim(1 - (avg_data[0] - 1) * 0.2, 1 + (avg_data[0] - 1) * 1.8)  # 放大显示前三个点
    ax1.set_xlim(-2, 52)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # 设置下图（完整范围）
    ax2.set_ylim(0.4, 1.1)
    ax2.set_xlim(-2, 52)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=1.0, color='gray', linestyle=':', alpha=0.7, linewidth=2)
    
    # 隐藏上图的下边框和下图的上述框，制造断轴效果
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # 不在上图显示x轴标签
    ax2.xaxis.tick_bottom()
    
    # 添加断轴标记
    d = 0.015  # 断线的大小
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)        # 左上断线
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # 右上断线
    
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # 左下断线
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # 右下断线
    
    # 设置标签
    ax2.set_xlabel('Quantization Parameter (QP)', fontsize=28)
    # ax1.set_ylabel('Normalized Quality $f_i(qp)$', fontsize=24)
    ax2.set_ylabel('Normalized Quality $f_i$', fontsize=28)
    
    ax1.tick_params(axis='both', which='major', labelsize=28)
    ax2.tick_params(axis='both', which='major', labelsize=28)
    
    # 添加图例
    ax1.legend(fontsize=28, framealpha=0.9, loc='upper right')
    
    # 添加标题
    # ax1.text(0.5, 1.05, f'Normalized Quality Function for {attr}', 
    #          transform=ax1.transAxes, fontsize=24, fontweight='bold', 
            #  ha='center')
    
    # 添加区域标注
    ax1.text(15, 1 + (avg_data[0] - 1), 'Detail View: QP=0,10,20', fontsize=28, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax2.text(10, 0.55, 'Full Range: QP=0-50', fontsize=28,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))
    
    plt.tight_layout()
    
    # 保存单个属性的图表
    filename = f"{attr.lower()}_quality_analysis.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    
    # 也保存PDF版本
    filename_pdf = f"{attr.lower()}_quality_analysis.pdf"
    filepath_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(filepath_pdf, dpi=300, bbox_inches='tight')
    
    plt.close()  # 关闭当前图表，准备绘制下一个
    
    print(f"已保存 {attr} 的图表到 {filepath}")

# 打印统计信息
print("\n平均值和标准差统计:")
for attr in attributes:
    print(f"\n{attr}:")
    print(f"QP levels: {qp_levels}")
    print(f"Average: {average_quality[attr]}")
    print(f"Std: {std_quality[attr]}")

print(f"\n所有图表已保存到文件夹: {output_dir}")