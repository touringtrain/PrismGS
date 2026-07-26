import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.interpolate import interp1d


def calculate_bitrate(config):
    """计算配置的总码率"""
    
    bitrate_data = {
        'position_high': {0: 49.8, 10: 49.8, 20: 49.8, 30: 49.8, 40: 49.8, 50: 49.8},
        'position_low': {0: 167.9, 10: 140.0, 20: 96.0, 30: 64.0, 40: 38.0, 50: 8.0},
        'SH_color': {0: 36.0, 10: 30.0, 20: 14.0, 30: 8.0, 40: 3.0, 50: 1.0},
        'opacity': {0: 54.0, 10: 42.0, 20: 19.0, 30: 8.0, 40: 2.4, 50: 1.0},
        'scale': {0: 59.0, 10: 48.0, 20: 21.0, 30: 11.5, 40: 7.0, 50: 1.0},
        'rotation': {0: 55.0, 10: 38.0, 20: 18.0, 30: 9.0, 40: 4.0, 50: 1.0}
    }
    
    channel_multipliers = {
        'position_high': 3, 'position_low': 3, 'SH_color': 3, 
        'opacity': 1, 'scale': 3, 'rotation': 4
    }
    
    config_mapping = {
        'position_low': 'position_low', 'color': 'SH_color', 
        'opacity': 'opacity', 'scale': 'scale', 'rotation': 'rotation'
    }
    
    bitrate_functions = {}
    
    qp_levels = np.array([0, 10, 20, 30, 40, 50])
    
    for attr, bitrates in bitrate_data.items():
        bitrate_points = np.array([bitrates[qp] for qp in qp_levels])
        bitrate_functions[attr] = interp1d(qp_levels, bitrate_points, 
                                                kind='linear', fill_value="extrapolate")
    
    total_bitrate = 49.8 * channel_multipliers['position_high']  # position_high固定
    
    for config_key, data_key in config_mapping.items():
        qp_value = config[config_key]
        attribute_bitrate = bitrate_functions[data_key](qp_value) * channel_multipliers[data_key]
        total_bitrate += attribute_bitrate
    
    return total_bitrate

def plot_combined_pareto_frontier(real_pareto_front, predicted_pareto_front, save_path=None):
    """绘制合并的Pareto前沿图，左侧纵轴为PSNR，右侧纵轴为差值"""
    
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    # 提取数据
    real_bitrates = [point[1] for point in real_pareto_front]
    real_qualities = [point[0] for point in real_pareto_front]
    
    predicted_bitrates = [point[1] for point in predicted_pareto_front]
    predicted_qualities = [point[0] for point in predicted_pareto_front]
    
    real_bitrates = np.array(real_bitrates) * 30 / 1000  # 转换为MB/s
    predicted_bitrates = np.array(predicted_bitrates) * 30 / 1000  # 转换为MB/s
    
    
    # 在主坐标轴（左侧）绘制Pareto前沿
    line1 = ax1.plot(real_bitrates, real_qualities, 'ro-', linewidth=0, markersize=7, markevery=3,
                     label='Real Pareto Frontier')
    line2 = ax1.plot(predicted_bitrates, predicted_qualities, 'bs-', linewidth=3, markersize=0, 
                     label='Predicted Pareto Frontier')
    
    # 设置主坐标轴
    ax1.set_xlabel('Bitrate (MB/s)', fontsize=18)
    ax1.set_ylabel('Quality (PSNR dB)', fontsize=18, color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
    ax1.tick_params(axis='x', labelsize=18)
    ax1.grid(True, alpha=0.3)
    
    # 创建次坐标轴（右侧）用于差值
    ax2 = ax1.twinx()
    
    # 计算差值
    diff = []
    diff_bitrates = []
    for rb, rq in zip(real_bitrates, real_qualities):
        # 找到预测前沿中最接近的码率点
        closest_idx = min(range(len(predicted_bitrates)), 
                         key=lambda i: abs(predicted_bitrates[i] - rb))
        pq = predicted_qualities[closest_idx]
        diff.append(rq - pq)
        diff_bitrates.append(rb)
    
    # 在次坐标轴（右侧）绘制差值
    line3 = ax2.plot(diff_bitrates, diff, 'mo--', linewidth=2, markersize=0, 
                     label='Difference (Real - Predicted)')
    ax2.axhline(0, color='gray', linestyle='--', linewidth=2)
    
    # 设置次坐标轴
    ax2.set_ylabel('PSNR Difference (dB)', fontsize=18, color='m')
    ax2.tick_params(axis='y', labelcolor='m', labelsize=18)
    
    # 合并图例
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, bbox_to_anchor=(0.26, 0.69), loc='upper left', borderaxespad=0, fontsize=18)
    
    # plt.title('Real vs Predicted Pareto Frontiers', fontsize=22)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"合并的Pareto前沿图已保存到: {save_path}")

if __name__ == "__main__":
    # 读取测试结果文件获取Pareto前沿
    import json
    
    pareto_front = []
    with open("../4k_1_bandwidth/test_results.json", "r") as f:
        test_results = json.load(f)
        for result in test_results:
            if result["bitrate_kb"] < 200:
                continue
            pareto_front.append([
                result["psnr"],
                result["bitrate_kb"]
            ])
    
    predicted_pareto_front = []
    with open("../4k_1_bandwidth/pareto_frontier_dp_4k_1.json", "r") as f:
        predicted_data = json.load(f)
        predicted_pareto_front = []
        for item in predicted_data:
            if item["bitrate_kb"] < 200:
                continue
            predicted_pareto_front.append([
                item["quality_psnr"],
                item["bitrate_kb"]
            ])

    # 绘制合并的图
    plot_combined_pareto_frontier(pareto_front, predicted_pareto_front, 
                                 save_path="method_combined_pareto_frontier.pdf")