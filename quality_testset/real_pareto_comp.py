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

def plot_pareto_frontier(real_pareto_front, predicted_pareto_front, save_path=None):
    """绘制Pareto前沿"""
    if not pareto_front:
        print("没有Pareto前沿数据可绘制")
        return
    
    real_bitrates = [point[1] for point in real_pareto_front]
    real_qualities = [point[0] for point in real_pareto_front]
    
    predicted_bitrates = [point[1] for point in predicted_pareto_front]
    predicted_qualities = [point[0] for point in predicted_pareto_front]
    
    plt.figure(figsize=(8, 6))
    plt.plot(real_bitrates, real_qualities, 'bo-', linewidth=2, markersize=3, label='Real Pareto Frontier')
    plt.plot(predicted_bitrates, predicted_qualities, 'gs--', linewidth=2, markersize=3, label='Predicted Pareto Frontier')
    
    
    unified_qp_quality = np.array([
        36.2693, 36.2334, 36.0995, 35.7345, 34.7456, 
        32.3418, 27.9024, 22.3123, 17.4712, 14.8614
    ])
    unified_qp_levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])
    
    unified_quality_func = CubicSpline(unified_qp_levels, unified_qp_quality)
    unified_qp_test = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    unified_qualities = [unified_quality_func(qp) for qp in unified_qp_test]
    unified_bitrates = [calculate_bitrate({attr: qp for attr in ['position_low', 'color', 'opacity', 'scale', 'rotation']}) 
                        for qp in unified_qp_test]
    
    # plt.plot(unified_bitrates, unified_qualities, 'ro--', linewidth=1, markersize=4, 
            # label='Uniform QP (Reference)')
    plt.xlabel('Bitrate (KB/frame)', fontsize=22)
    plt.ylabel('Quality (PSNR dB)', fontsize=22)
    plt.title('PSNR of Real and Predicted Pareto Frontiers', fontsize=22)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto前沿图已保存到: {save_path}")
        
def plot_diff(real_pareto_front, predicted_pareto_front, save_path=None):
    """绘制真实与预测Pareto前沿的差异，纵轴为差值"""
    real_bitrates = [point[1] for point in real_pareto_front]
    real_qualities = [point[0] for point in real_pareto_front]
    
    predicted_bitrates = [point[1] for point in predicted_pareto_front]
    predicted_qualities = [point[0] for point in predicted_pareto_front]
    
    diff = []
    for rb, rq in zip(real_bitrates, real_qualities):
        # 找到预测前沿中最接近的码率点
        closest_idx = min(range(len(predicted_bitrates)), key=lambda i: abs(predicted_bitrates[i] - rb))
        pq = predicted_qualities[closest_idx]
        diff.append(rq - pq)
        
    plt.figure(figsize=(8, 6))
    plt.plot(real_bitrates, diff, 'mo-', linewidth=2, markersize=3, label='Difference (Real - Predicted)')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Bitrate (KB/frame)', fontsize=22)
    plt.ylabel('Quality Difference (PSNR dB)', fontsize=22)
    plt.title('PSNR Difference between Real and Predicted Pareto Frontiers', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=22)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
if __name__ == "__main__":
    # read the test_results.json file to get the pareto frontier
    import json
    pareto_front = []
    with open("test_results.json", "r") as f:
        test_results = json.load(f)
        for result in test_results:
            if result["bitrate_kb"] < 290:
                continue
            pareto_front.append([
                result["psnr"],
                result["bitrate_kb"]
            ])
    
    predicted_pareto_front = []
    with open("../pareto_frontier_dp.json", "r") as f:
        predicted_data = json.load(f)
        predicted_pareto_front = []
        for item in predicted_data:
            if item["bitrate_kb"] < 290:
                continue
            predicted_pareto_front.append([
                item["quality_psnr"],
                item["bitrate_kb"]
            ])

    # plot_pareto_frontier(pareto_front, predicted_pareto_front, save_path="pareto_frontier.png")
    plot_pareto_frontier(pareto_front, predicted_pareto_front, save_path="method_pareto_frontier.pdf")
    
    plot_diff(pareto_front, predicted_pareto_front, save_path="method_pareto_frontier_diff.pdf")