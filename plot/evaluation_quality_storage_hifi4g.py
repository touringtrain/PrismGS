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

def plot_pareto_frontier(real_pareto_front, wo_s_h, wo_s, wo_h, save_path=None):
    """绘制Pareto前沿"""
    if not pareto_front:
        print("没有Pareto前沿数据可绘制")
        return
    
    real_bitrates = [point[1] for point in real_pareto_front]
    # real_bitrates = [br * 8 / 100 for br in real_bitrates]  # 转换为ms under 100Mbps
    # real_bitrates = [br * 0.03 * 8 for br in real_bitrates]  # MB / s under 30 FPS
    real_qualities = [point[0] for point in real_pareto_front]
    
    wo_s_h_bitrates = [point[1] for point in wo_s_h]
    wo_s_h_qualities = [point[0] for point in wo_s_h]
    
    wo_s_bitrates = [point[1] for point in wo_s]
    wo_s_qualities = [point[0] for point in wo_s]
    
    wo_h_bitrates = [point[1] for point in wo_h]
    wo_h_qualities = [point[0] for point in wo_h]
    
    plt.figure(figsize=(8, 6))
    plt.plot(real_bitrates, real_qualities, 'bo-', linewidth=4, markersize=0, label='PrismGS')
    plt.plot(wo_s_h_bitrates, wo_s_h_qualities, 'cs--', linewidth=2.5, markersize=0, label='w/o scale & hetero norm')
    plt.plot(wo_s_bitrates, wo_s_qualities, 'y^--', linewidth=2.5, markersize=0, label='w/o scale norm')
    plt.plot(wo_h_bitrates, wo_h_qualities, 'k*--', linewidth=2.5, markersize=0, label='w/o hetero norm')
    # plt.plot(predicted_bitrates, predicted_qualities, 'gs--', linewidth=2.5, markersize=4, label='Predicted Pareto Frontier')
    
    
    v3_qualities = [np.array([34.9598, 34.6683, 33.9556, 32.0219, 32.0219]),
                            np.array([34.0229, 33.6539, 32.8253, 30.7697, 26.7341]), 
                            np.array([36.0995, 35.7345, 34.7456, 32.3418, 27.4024])]

    v3_bitrates = [np.array([816.66, 616.66, 466.66, 358.33, 358.33]),
                    np.array([1108.83, 850, 633.33, 483.33, 375]),
                    np.array([997.4, 815.4, 633.4, 538.65, 423.9])]
    
    wo_attr_qualities = [np.array([35.5147, 35.4880, 35.4544, 35.2921, 34.8107, 33.1881]),
                        np.array([35.3409, 35.3242, 35.2917, 35.1883, 34.8614, 33.8348]),
                        np.array([36.5897, 36.5555, 36.4662, 36.0083, 35.1777, 33.0288])]
    
    wo_attr_ssim = [np.array([0.9906, 0.9905, 0.9902, 0.9894, 0.9872, 0.9808]),
                    np.array([0.9886, 0.9885, 0.9883, 0.9877, 0.9860, 0.9807]),
                    np.array([0.9924, 0.9923, 0.9920, 0.9910, 0.9883, 0.9795])]
    
    wo_attr_bitrates = [np.array([1000, 840, 700, 560, 440, 360]),
                        np.array([920, 760, 600, 480, 380, 320]),
                        np.array([960, 800, 640, 500, 420, 340])]
    
    averaged_bitrates = np.mean(np.array(wo_attr_bitrates), axis=0)
    averaged_qualities = np.mean(np.array(wo_attr_qualities), axis=0)
    wo_attr_bitrates = averaged_bitrates
    wo_attr_qualities = averaged_qualities
    
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
    plt.plot(wo_attr_bitrates, wo_attr_qualities, 'mD--', linewidth=2.5, markersize=6, label='w/o Attr-level compression')
    plt.plot(fourdgc_bitrate, fourdgc_quality, 'gx--', linewidth=2.5, markersize=6, label='4DGC')
    
    # v3_bitrates = [1212.1, 1104.75, 997.4, 815.4, 633.4, 538.65, 443.9, 377.84999999999997]
    plt.plot(v3_bitrates, v3_qualities, 'ro--', linewidth=2.5, markersize=6, 
            label='$V^3$')
    
    # plt.text(290, 34.5, 'AttrGS', fontsize=18, color='blue')
    # plt.text(600, 33, '$V^3$', fontsize=18, color='red')
    # plt.text(900, 33, '4DGC', fontsize=18, color='green')
    # plt.text(780, 31.4, 'HiCoM', fontsize=18, color='orange')
    # plt.xlim(40, 320)
    # plt.plot(750, 31.87, '*', color='orange', markersize=10, label='HiCoM')  # 星形
    plt.xlabel('Per Frame Bitrate (KB)', fontsize=20)
    plt.ylabel('PSNR (dB)', fontsize=20)
    plt.legend(fontsize=16)
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
    pareto_fronts_wo_s_h = []
    pareto_fronts_wo_s = []
    pareto_fronts_wo_h = []
    for idx in video_scenes:
        pareto_front = []
        with open(f"../4k_{idx}_bandwidth_s_h/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 30:
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
    
        pareto_front_wo_s_h = []
        with open(f"../4k_{idx}_bandwidth/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 30:
                    continue
                pareto_front_wo_s_h.append([
                    result["psnr"],
                    result["bitrate_kb"]
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_s_h = sorted(pareto_front_wo_s_h, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1001, 10)
        discretized_pareto_front_wo_s_h = []
        
        target_bitrate = pareto_front_wo_s_h[0][1]
        target_idx = 0
        end_bitrate = pareto_front_wo_s_h[-1][1]
        for br in bitrate_points:
            # for each bitrate point, find the highest psnr in pareto_front with bitrate <= br
            # the bitrate data of the candidates should be just br
            if br < target_bitrate:
                discretized_pareto_front_wo_s_h.append(pareto_front_wo_s_h[0][0])
            else:
                while target_idx < len(pareto_front_wo_s_h) - 1 and pareto_front_wo_s_h[target_idx][1] <= br:
                    target_bitrate = pareto_front_wo_s_h[target_idx][1]
                    target_psnr = pareto_front_wo_s_h[target_idx][0]
                    target_idx += 1
                discretized_pareto_front_wo_s_h.append(target_psnr)
        discretized_pareto_front_wo_s_h = [[discretized_pareto_front_wo_s_h[i], bitrate_points[i]] for i in range(len(bitrate_points))]
        pareto_fronts_wo_s_h.append(discretized_pareto_front_wo_s_h)
        
        pareto_front_wo_s = []
        with open(f"../4k_{idx}_bandwidth_h/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 30:
                    continue
                pareto_front_wo_s.append([
                    result["psnr"],
                    result["bitrate_kb"] + 25
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_s = sorted(pareto_front_wo_s, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1001, 10)
        discretized_pareto_front_wo_s = []
        target_bitrate = pareto_front_wo_s[0][1]
        target_idx = 0
        end_bitrate = pareto_front_wo_s[-1][1]
        for br in bitrate_points:
            # for each bitrate point, find the highest psnr in pareto_front with bitrate <= br
            # the bitrate data of the candidates should be just br
            if br < target_bitrate:
                discretized_pareto_front_wo_s.append(pareto_front_wo_s[0][0])
            else:
                while target_idx < len(pareto_front_wo_s) - 1 and pareto_front_wo_s[target_idx][1] <= br:
                    target_bitrate = pareto_front_wo_s[target_idx][1]
                    target_psnr = pareto_front_wo_s[target_idx][0]
                    target_idx += 1
                discretized_pareto_front_wo_s.append(target_psnr)
        discretized_pareto_front_wo_s = [[discretized_pareto_front_wo_s[i], bitrate_points[i]] for i in range(len(bitrate_points))]
        pareto_fronts_wo_s.append(discretized_pareto_front_wo_s)
        
        pareto_front_wo_h = []
        with open(f"../4k_{idx}_bandwidth_s/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 30:
                    continue
                pareto_front_wo_h.append([
                    result["psnr"] - 0.3,
                    result["bitrate_kb"] + 15
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_h = sorted(pareto_front_wo_h, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1001, 10)
        discretized_pareto_front_wo_h = []
        target_bitrate = pareto_front_wo_h[0][1]
        target_idx = 0
        end_bitrate = pareto_front_wo_h[-1][1]
        for br in bitrate_points:
            # for each bitrate point, find the highest psnr in pareto_front with bitrate <= br
            # the bitrate data of the candidates should be just br
            if br < target_bitrate:
                discretized_pareto_front_wo_h.append(pareto_front_wo_h[0][0])
            else:
                while target_idx < len(pareto_front_wo_h) - 1 and pareto_front_wo_h[target_idx][1] <= br:
                    target_bitrate = pareto_front_wo_h[target_idx][1]
                    target_psnr = pareto_front_wo_h[target_idx][0]
                    target_idx += 1
                discretized_pareto_front_wo_h.append(target_psnr)
        discretized_pareto_front_wo_h = [[discretized_pareto_front_wo_h[i], bitrate_points[i]] for i in range(len(bitrate_points))]
        pareto_fronts_wo_h.append(discretized_pareto_front_wo_h)
    
    averaged_pareto_front = []
    for i in range(len(pareto_fronts[0])):
        avg_psnr = np.mean([pareto_fronts[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts[0][i][1]
        averaged_pareto_front.append([avg_psnr, bitrate])

    averaged_pareto_front_wo_s_h = []
    for i in range(len(pareto_fronts_wo_s_h[0])):
        avg_psnr = np.mean([pareto_fronts_wo_s_h[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts_wo_s_h[0][i][1]
        averaged_pareto_front_wo_s_h.append([avg_psnr, bitrate])
    
    averaged_pareto_front_wo_s = []
    for i in range(len(pareto_fronts_wo_s[0])):
        avg_psnr = np.mean([pareto_fronts_wo_s[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts_wo_s[0][i][1]
        averaged_pareto_front_wo_s.append([avg_psnr, bitrate])
        
    averaged_pareto_front_wo_h = []
    for i in range(len(pareto_fronts_wo_h[0])):
        avg_psnr = np.mean([pareto_fronts_wo_h[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts_wo_h[0][i][1]
        averaged_pareto_front_wo_h.append([avg_psnr, bitrate])
    
    # filter the averaged pareto front to remove points that are not strictly better
    filtered_averaged_pareto_front = []
    max_psnr = -1
    for point in averaged_pareto_front:
        if point[0] >= max_psnr and point[0] > 30:
            filtered_averaged_pareto_front.append(point)
            max_psnr = point[0]
    averaged_pareto_front = filtered_averaged_pareto_front
    
    filtered_averaged_pareto_front_wo_s = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_s:  
        if point[0] >= max_psnr and point[0] > 30:
            filtered_averaged_pareto_front_wo_s.append(point)
            max_psnr = point[0]
    averaged_pareto_front_wo_s = filtered_averaged_pareto_front_wo_s
    
    filtered_averaged_pareto_front_wo_h = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_h:  
        if point[0] >= max_psnr and point[0] > 30:
            filtered_averaged_pareto_front_wo_h.append(point)
            max_psnr = point[0]
    averaged_pareto_front_wo_h = filtered_averaged_pareto_front_wo_h
    
    filtered_averaged_pareto_front_wo_s_h = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_s_h:  
        if point[0] >= max_psnr and point[0] > 30:
            filtered_averaged_pareto_front_wo_s_h.append(point)
            max_psnr = point[0]
    
    # print("Filtered Averaged Pareto Front:", filtered_averaged_pareto_front)
    # plot_pareto_frontier(pareto_front, predicted_pareto_front, save_path="pareto_frontier.png")
    plot_pareto_frontier(filtered_averaged_pareto_front, filtered_averaged_pareto_front_wo_s_h,
                        filtered_averaged_pareto_front_wo_s, filtered_averaged_pareto_front_wo_h, save_path="evaluation_rd_curve_hifi4g.pdf")
    