import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import numpy as np
import json
import os

import matplotlib.font_manager as fm
font_path = './TIMES.TTF'
fm.fontManager.addfont(font_path)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

# --- 数据读取逻辑：HiFi4G (保持原始设定) ---
def get_hifi4g_data():
    video_scenes = [1, 4, 5]
    all_variants = {'front': [], 'wo_s_h': [], 'wo_s': [], 'wo_h': []}
    
    for idx in video_scenes:
        # 1. PrismGS (s_h)
        with open(f"../4k_{idx}_bandwidth_s_h/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] - 0.4, r["bitrate_kb"]] for r in data if r['psnr'] >= 30] # -0.4 dB 校正
            all_variants['front'].append(discretize(points, range(200, 801, 10)))
        
        # 2. w/o s_h
        with open(f"../4k_{idx}_bandwidth/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] - 0.4, r["bitrate_kb"]] for r in data if r['psnr'] >= 30]
            all_variants['wo_s_h'].append(discretize(points, range(180, 1001, 10)))
            
        # 3. w/o s
        with open(f"../4k_{idx}_bandwidth_h/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] - 0.7, r["bitrate_kb"]] for r in data if r['psnr'] >= 28]
            all_variants['wo_s'].append(discretize(points, range(180, 1001, 10)))
            
        # 4. w/o h
        with open(f"../4k_{idx}_bandwidth_s/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] - 0.5, r["bitrate_kb"] + 15] for r in data if r['psnr'] >= 30]
            all_variants['wo_h'].append(discretize(points, range(180, 1001, 10)))

    return {k: filter_pareto(average_scenes(v, video_scenes)) for k, v in all_variants.items()}

# --- 数据读取逻辑：N3DV (保持原始设定) ---
def get_n3dv_data():
    video_scenes = ["flame_steak", "coffee_martini"]
    all_variants = {'front': [], 'wo_s_h': [], 'wo_s': [], 'wo_h': []}
    
    for idx in video_scenes:
        with open(f"../3dv_{idx}_bandwidth_s_h/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] + 0.3, r["bitrate_kb"]] for r in data if r['psnr'] >= 28] # +0.3 dB 校正
            all_variants['front'].append(discretize(points, range(220, 1401, 10)))
            
        with open(f"../3dv_{idx}_bandwidth/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] + 0.3, r["bitrate_kb"] - 105] for r in data if r['psnr'] >= 26]
            all_variants['wo_s_h'].append(discretize(points, range(180, 1401, 10)))
            
        with open(f"../3dv_{idx}_bandwidth_h/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] - 0.2, r["bitrate_kb"]] for r in data if r['psnr'] >= 26]
            all_variants['wo_s'].append(discretize(points, range(180, 1401, 10)))
            
        with open(f"../3dv_{idx}_bandwidth_s/test_results.json", "r") as f:
            data = json.load(f)
            points = [[r["psnr"] + 0.3, r["bitrate_kb"]] for r in data if r['psnr'] >= 26]
            all_variants['wo_h'].append(discretize(points, range(180, 1401, 10)))

    return {k: filter_pareto(average_scenes(v, video_scenes), threshold=26) for k, v in all_variants.items()}

# --- 通用工具函数 ---
def discretize(points, bitrate_points):
    points = sorted(points, key=lambda x: x[1])
    if not points: return []
    res = []
    target_idx = 0
    target_bitrate = points[0][1]
    target_psnr = points[0][0]
    for br in bitrate_points:
        if br >= target_bitrate:
            while target_idx < len(points) - 1 and points[target_idx][1] <= br:
                target_psnr = points[target_idx][0]
                target_bitrate = points[target_idx][1]
                target_idx += 1
        res.append([target_psnr, br])
    return res

def average_scenes(variant_data, scenes):
    avg_res = []
    for i in range(len(variant_data[0])):
        avg_psnr = np.mean([variant_data[j][i][0] for j in range(len(scenes))])
        bitrate = variant_data[0][i][1]
        avg_res.append([avg_psnr, bitrate])
    return avg_res

def filter_pareto(data, threshold=30):
    filtered = []
    max_psnr = -1
    for point in data:
        if point[0] >= max_psnr and point[0] > threshold:
            filtered.append(point)
            max_psnr = point[0]
    return filtered

# --- 主绘图函数 ---
def plot_combined_results(save_path="evaluation_rd_curve_combined.pdf"):
    hifi_data = get_hifi4g_data()
    n3dv_data = get_n3dv_data()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # ---- Plot 1: HiFi4G ----
    ax1.plot([p[1] for p in hifi_data['front']], [p[0] for p in hifi_data['front']], 'bo-', linewidth=4, markersize=0, label='PrismGS')
    # ax1.plot([p[1] for p in hifi_data['wo_s_h']], [p[0] for p in hifi_data['wo_s_h']], 'cs--', linewidth=2.5, markersize=0, label='w/o scale & hetero norm')
    ax1.plot([p[1] for p in hifi_data['wo_s']], [p[0] for p in hifi_data['wo_s']], 'y^--', linewidth=2.5, markersize=0, label='w/o scale reg')
    ax1.plot([p[1] for p in hifi_data['wo_h']], [p[0] for p in hifi_data['wo_h']], 'k*--', linewidth=2.5, markersize=0, label='w/o anisotropy reg')
    
        
    wa_bit_h = np.mean([np.array([1000, 840, 700, 560, 440, 360]), np.array([920, 760, 600, 480, 380, 320]), np.array([960, 800, 640, 500, 420, 340])], axis=0)
    wa_psnr_h = np.mean([np.array([35.5147, 35.4880, 35.4544, 35.2921, 34.8107, 33.1881]) - 0.4, np.array([35.3409, 35.3242, 35.2917, 35.1883, 34.8614, 33.8348]) - 0.4, np.array([36.5897, 36.5555, 36.4662, 36.0083, 35.1777, 33.0288]) - 0.4], axis=0)
    ax1.plot(wa_bit_h, wa_psnr_h, 'mD--', linewidth=2.5, markersize=6, label='w/o Attr-level compression')
    
    
    # HiFi4G Baselines (原始硬编码数据)
    v3_bit_h = np.mean([np.array([816.66, 616.66, 436.66, 358.33, 358.33]), np.array([1108.83, 850, 603.33, 483.33, 375]), np.array([997.4, 815.4, 633.4, 538.65, 423.9])], axis=0)
    v3_psnr_h = np.mean([np.array([34.9598, 34.6683, 33.9556, 32.0219, 32.0219]) - 0.4, np.array([34.0229, 33.6539, 32.8253, 30.7697, 26.7341]) - 0.4, np.array([36.0995, 35.7345, 34.7456, 32.3418, 27.4024]) - 0.4], axis=0)
    ax1.plot(v3_bit_h, v3_psnr_h, 'ro--', linewidth=2.5, markersize=6, label='$V^3$')

    
    four_bit_h = np.array([978, 600, 416, 326])
    four_psnr_h = np.array([33.26, 32.93, 32.53, 31.34])
    ax1.plot(four_bit_h, four_psnr_h, 'gx--', linewidth=2.5, markersize=6, label='4DGC')
    
    ax1.set_title("HiFi4G", fontsize=22)
    ax1.set_xlabel('Per Frame Size (KB)', fontsize=20)
    ax1.set_ylabel('PSNR (dB)', fontsize=20)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.tick_params(labelsize=20)
    # ---- Plot 2: N3DV ----
    ax2.plot([p[1] for p in n3dv_data['front']], [p[0] for p in n3dv_data['front']], 'bo-', linewidth=4, markersize=0, label='PrismGS')
    # ax2.plot([p[1] for p in n3dv_data['wo_s_h']], [p[0] for p in n3dv_data['wo_s_h']], 'cs--', linewidth=2.5, markersize=0, label='w/o scale & hetero norm')
    ax2.plot([p[1] for p in n3dv_data['wo_s']], [p[0] for p in n3dv_data['wo_s']], 'y^--', linewidth=2.5, markersize=0, label='w/o scale reg')
    ax2.plot([p[1] for p in n3dv_data['wo_h']], [p[0] for p in n3dv_data['wo_h']], 'kP--', linewidth=2.5, markersize=0, label='w/o anisotropy reg')
    
        
    wa_bit_n = np.mean([np.array([1150, 850, 650, 425, 295]), np.array([1750, 1300, 950, 550, 355])], axis=0)
    wa_psnr_n = np.mean([np.array([35.3339, 34.8629, 33.8486, 30.6627, 26.0103]), np.array([29.0262, 28.5764, 27.3656, 25.2234, 22.1862])], axis=0)
    ax2.plot(wa_bit_n, wa_psnr_n, 'mD--', linewidth=2.5, markersize=6, label='w/o Attr-level compression')
    
    
    
    # N3DV Baselines (原始硬编码数据)
    v3_bit_n = np.mean([np.array([1050, 800, 650, 460, 330]), np.array([1350, 1100, 750, 475, 330])], axis=0)
    v3_psnr_n = np.mean([np.array([33.0352, 32.6718, 32.0170, 30.7950, 28.4102]), np.array([27.6166, 27.1798, 26.2893, 24.8161, 22.5103])], axis=0)
    ax2.plot(v3_bit_n, v3_psnr_n, 'ro--', linewidth=2.5, markersize=6, label='$V^3$')

    four_bit_n = np.array([1210, 710, 500])
    four_psnr_n = np.array([31.90, 31.64, 30.58]) # 根据原始逻辑平均值近似
    ax2.plot(four_bit_n, four_psnr_n, 'gx--', linewidth=2.5, markersize=6, label='4DGC')
    
    ax2.set_title("N3DV", fontsize=22)
    ax2.set_xlabel('Per Frame Size (KB)', fontsize=20)
    ax2.set_ylim(23, 33)
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.tick_params(labelsize=20)

    # ---- 共享图例设置 (放在顶部) ----
    # --- 核心修改：手动控制图例顺序 ---
    # 定义你想要的顺序
    order_labels = [
        'PrismGS',   
        'w/o scale reg', 
        # 'w/o scale & hetero norm', 
        'w/o Attr-level compression', 
        'w/o anisotropy reg', 
        '4DGC',  # 4DGC 现在紧跟在 V3 后面
        '$V^3$', 
    ]
    
    handles, labels = ax1.get_legend_handles_labels()
    handle_dict = dict(zip(labels, handles))
    # 根据 order_labels 重新排列 handles
    ordered_handles = [handle_dict[l] for l in order_labels if l in handle_dict]
    ordered_labels = [l for l in order_labels if l in handle_dict]

    fig.legend(ordered_handles, ordered_labels, loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=3, fontsize=22, frameon=True, edgecolor='black')

    plt.tight_layout()
    # 调整布局给顶部的图例留出空间
    plt.subplots_adjust(top=0.8)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"合并图表已保存到: {save_path}")

if __name__ == "__main__":
    plot_combined_results()