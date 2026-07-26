import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import matplotlib.font_manager as fm

# 注册新字体
font_path = './TIMES.TTF'
fm.fontManager.addfont(font_path)

# --- 基础配置 ---
attributes = [r'$\mu_{high}$', r'$\mu_{low}$', r'$\mathbf{c}$', r'$\alpha$', r'$\mathbf{s}$', r'$\mathbf{R}$']
colors = ['#66B2FF', '#DAE8FC', '#D5E8D4', '#FFE6CC', '#F8CECC', '#E1D5E7']

# 学术字体设置（嵌入 Type 42 字体）
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# times new roman 字体设置
plt.rcParams['font.family'] = 'Times New Roman'

plt.rcParams['mathtext.fontset'] = 'stix'

# 数据字典
scenes_data = {
    '4k_1_Greeting': {
        'low': [125, 14, 36, 8, 38, 76],
        'high': [125, 193, 130, 22, 38, 180],
        'labels': ['300 KB', '700 KB'],
    },
    'Coffee_Martini': {
        'low': [48, 142, 79, 29, 102, 96],
        'high': [48, 320, 225, 48, 175, 435],
        'labels': ['500 KB', '1200 KB'],
    }
}

def save_individual_plot(scene_name, data_info):
    # 创建独立画布，调整尺寸以适应单图比例
    fig, ax = plt.subplots(figsize=(7, 2))
    
    x = np.arange(len(attributes))
    width = 0.35
    
    data_low = data_info['low']
    data_high = data_info['high']
    labels = data_info['labels']
    
    # 绘制柱状图
    for i in range(len(attributes)):
        # Low Budget
        ax.bar(x[i] - width/2, data_low[i], width, color=colors[i], 
               edgecolor='black', linewidth=2, alpha=0.9)
        # High Budget
        ax.bar(x[i] + width/2, data_high[i], width, color=colors[i], 
               edgecolor='black', linewidth=2, hatch='//', alpha=0.7)

    # 坐标轴与标题
    ax.set_xticks(x)
    ax.set_xticklabels(attributes, fontsize=16)
    ax.set_ylabel('Size (KB / frame)', fontsize=16)

    # ax.set_title(data_info['title'], fontweight='bold', pad=12, fontsize=12)
    
    # font of the ticks
    ax.tick_params(axis='y', labelsize=16)


    # 网格与边框
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 标注说明 (位置根据你修改后的 0.22 保持一致)
    # ax.text(0.78, 0.9, f'Solid: {labels[0]}\nHatched: {labels[1]}', 
    #         transform=ax.transAxes, ha='right', va='top', 
            # fontsize=14, bbox=dict(boxstyle="round", facecolor='grey', alpha=0.15, edgecolor='none'))

    # 图例（每张图单独带一个图例，或者你可以注释掉这部分在 LaTeX 中统一写）
    # legend_elements = [Patch(facecolor=colors[i], label=attributes[i]) for i in range(len(attributes))]
    # ax.legend(handles=legend_elements, loc='upper right', ncol=2, 
    #           frameon=False, fontsize=14, columnspacing=0.8)

    plt.tight_layout()
    # 保存文件
    filename = f'evaluation_rate_allocation_{scene_name.lower()}.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)

# --- 执行循环保存 ---
for scene, info in scenes_data.items():
    save_individual_plot(scene, info)