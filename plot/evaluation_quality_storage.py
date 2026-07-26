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

def plot_pareto_frontier(real_pareto_front, save_path=None):
    """绘制Pareto前沿"""
    if not pareto_front:
        print("没有Pareto前沿数据可绘制")
        return
    
    real_bitrates = [point[1] for point in real_pareto_front]
    # real_bitrates = [br * 8 / 100 for br in real_bitrates]  # 转换为ms under 100Mbps
    # real_bitrates = [br * 0.03 * 8 for br in real_bitrates]  # MB / s under 30 FPS
    real_qualities = [point[0] for point in real_pareto_front]
    
    plt.figure(figsize=(8, 6))
    plt.plot(real_bitrates, real_qualities, 'bo-', linewidth=4, markersize=0, label='Real Pareto Frontier')
    # plt.plot(predicted_bitrates, predicted_qualities, 'gs--', linewidth=2.5, markersize=4, label='Predicted Pareto Frontier')
    
    
    v3_qualities = [np.array([34.9598, 34.6683, 33.9556, 32.0219, 32.0219]),
                            np.array([34.0229, 33.6539, 32.8253, 30.7697, 26.7341]), 
                            np.array([36.0995, 35.7345, 34.7456, 32.3418, 27.4024])]

    v3_bitrates = [np.array([816.66, 616.66, 466.66, 358.33, 358.33]),
                    np.array([1108.83, 850, 633.33, 483.33, 375]),
                    np.array([997.4, 815.4, 633.4, 538.65, 423.9])]
    
    averaged_bitrates = np.mean(np.array(v3_bitrates), axis=0)
    averaged_qualities = np.mean(np.array(v3_qualities), axis=0)
    v3_bitrates = averaged_bitrates
    v3_qualities = averaged_qualities
    
    # v3_bitrates = [br * 0.03 * 8 for br in v3_bitrates]  # MB / s under 30 FPS
    
    fourdgc_quality = [np.array([32.66, 31.91, 30.41, 29.74]), # 1e-5, 3e-5, 5e-5, 1e-4
                np.array([32.66, 31.91, 30.41, 29.74]),
                np.array([32.66, 31.91, 30.41, 29.74])]

    fourdgc_bitrate = [np.array([1000, 500, 416, 376]),
                np.array([1000, 500, 416, 376]),
                np.array([1000, 500, 416, 376])]
    
    averaged_bitrates = np.mean(np.array(fourdgc_bitrate), axis=0)
    averaged_qualities = np.mean(np.array(fourdgc_quality), axis=0)
    fourdgc_bitrate = averaged_bitrates
    fourdgc_quality = averaged_qualities
    
    plt.plot(fourdgc_bitrate, fourdgc_quality, 'gx--', linewidth=2.5, markersize=6, label='4DGC')
    
    # v3_bitrates = [1212.1, 1104.75, 997.4, 815.4, 633.4, 538.65, 443.9, 377.84999999999997]
    plt.plot(v3_bitrates, v3_qualities, 'ro--', linewidth=2.5, markersize=6, 
            label='Uniform QP (Reference)')
    plt.text(290, 34.5, 'AttrGS', fontsize=18, color='blue')
    plt.text(600, 33, '$V^3$', fontsize=18, color='red')
    plt.text(900, 33, '4DGC', fontsize=18, color='green')
    plt.text(780, 31.4, 'HiCoM', fontsize=18, color='orange')
    # plt.xlim(40, 320)
    plt.plot(750, 31.87, '*', color='orange', markersize=10, label='HiCoM')  # 星形
    plt.xlabel('Per Frame Bitrate (KB)', fontsize=20)
    plt.ylabel('PSNR (dB)', fontsize=20)
    # plt.title('Visual Quality and Transmission Time under 100Mbps', fontsize=22)
    # plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto前沿图已保存到: {save_path}")
        
if __name__ == "__main__":
    # read the test_results.json file to get the pareto frontier
    import json
    video_scenes = [1, 4, 5]
    pareto_fronts = []
    for idx in video_scenes:
        pareto_front = []
        with open(f"../4k_{idx}_bandwidth/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result["bitrate_kb"] < 200 or result["bitrate_kb"] > 1200:
                    continue
                pareto_front.append([
                    result["psnr"],
                    result["bitrate_kb"]
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front = sorted(pareto_front, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(200, 801, 10)
        discretized_pareto_front = []
        
        target_bitrate = pareto_front[0][1]
        target_idx = 0
        end_bitrate = pareto_front[-1][1]
        for br in bitrate_points:
            # for each bitrate point, find the highest psnr in pareto_front with bitrate <= br
            # the bitrate data of the candidates should be just br
            if br < target_bitrate:
                discretized_pareto_front.append(pareto_front[0][0])
            else:
                while target_idx < len(pareto_front) - 1 and pareto_front[target_idx][1] <= br:
                    target_bitrate = pareto_front[target_idx][1]
                    target_psnr = pareto_front[target_idx][0]
                    target_idx += 1
                discretized_pareto_front.append(target_psnr)
        discretized_pareto_front = [[discretized_pareto_front[i], bitrate_points[i]] for i in range(len(bitrate_points))]
        pareto_fronts.append(discretized_pareto_front)
    
    averaged_pareto_front = []
    for i in range(len(pareto_fronts[0])):
        avg_psnr = np.mean([pareto_fronts[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts[0][i][1]
        averaged_pareto_front.append([avg_psnr, bitrate])
    # print(averaged_pareto_front)
    
    # filter the averaged pareto front to remove points that are not strictly better
    filtered_averaged_pareto_front = []
    max_psnr = -1
    for point in averaged_pareto_front:
        if point[0] >= max_psnr and point[0] > 28:
            filtered_averaged_pareto_front.append(point)
            max_psnr = point[0]
    averaged_pareto_front = filtered_averaged_pareto_front
    # print("Filtered Averaged Pareto Front:", filtered_averaged_pareto_front)
    # plot_pareto_frontier(pareto_front, predicted_pareto_front, save_path="pareto_frontier.png")
    plot_pareto_frontier(filtered_averaged_pareto_front, save_path="evaluation_rd_curve_hifi4g.pdf")
    