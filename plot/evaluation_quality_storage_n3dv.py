import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy as np
from scipy.interpolate import interp1d
import os

def calculate_bitrate(config):
    """计算配置的总码率"""
    
    bitrate_data = {}
        
    ablation_dict = {
        "position_high": [1, 3, 5],
        "position_low": [0, 2, 4],
        "color": [9, 10, 11],
        "opacity": [12],
        "scale": [13, 14, 15],
        "rotation": [16, 17, 18, 19]
    }
    
    qps = [0, 10, 20, 30, 40, 50]
    for attr, indices in ablation_dict.items():
        bitrate_data[attr] = {}
        for qp in qps:
            total_size = 0.0
            for i in indices:
                if attr == "position_high":
                    video_path = f"feature_video/png_all_0/group0/{i}.mp4"
                else:
                    video_path = f"feature_video/png_all_{qp}/group0/{i}.mp4"
                size_bytes = os.path.getsize(video_path)
                size_megabits = (size_bytes) / 20 / 1000
                total_size += size_megabits
            bitrate_data[attr][qp] = total_size
            bitrate_data[attr][qp] /= len(indices)  # 平均码率
    print("读取的码率数据:", bitrate_data)
    
    channel_multipliers = {
        'position_high': 3, 'position_low': 3, 'color': 3, 
        'opacity': 1, 'scale': 3, 'rotation': 4
    }
    
    config_mapping = {
        'position_low': 'position_low', 'color': 'color', 
        'opacity': 'opacity', 'scale': 'scale', 'rotation': 'rotation'
    }
    
    bitrate_functions = {}
    
    qp_levels = np.array([0, 10, 20, 30, 40, 50])
    
    for attr, bitrates in bitrate_data.items():
        bitrate_points = np.array([bitrates[qp] for qp in qp_levels])
        bitrate_functions[attr] = interp1d(qp_levels, bitrate_points, 
                                                kind='linear', fill_value="extrapolate")
    
    total_bitrate = 15 * channel_multipliers['position_high']  # position_high固定
    
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
    plt.plot(wo_h_bitrates, wo_h_qualities, 'kP--', linewidth=2.5, markersize=0, label='w/o hetero norm')
    # plt.plot(predicted_bitrates, predicted_qualities, 'gs--', linewidth=2.5, markersize=4, label='Predicted Pareto Frontier')
    
    
    v3_qualities = [np.array([33.0352, 32.9718, 32.0170, 30.7950, 28.4102]),
                            np.array([27.6166, 27.1798, 26.2893, 24.8161, 22.5103]), 
                            # np.array([33.0352, 32.9718, 32.0170, 30.7950, 28.4102]),
                            # np.array([30.1170, 29.9141, 29.7124, 28.8205, 27.9165])
                            ]

    v3_bitrates = [np.array([850, 800, 650, 460, 330]),
                    np.array([1050, 1020, 750, 475, 330]),
                    # np.array([850, 800, 650, 460, 330])
                 ]
    
    averaged_bitrates = np.mean(np.array(v3_bitrates), axis=0)
    averaged_qualities = np.mean(np.array(v3_qualities), axis=0)
    v3_bitrates = averaged_bitrates
    v3_qualities = averaged_qualities
    
    wo_attr_qualities = [np.array([35.3339, 34.8629, 33.8486, 30.6627, 26.0103]),
                        np.array([29.0262, 28.5764, 27.3656, 25.2234, 22.1862])]
    
    wo_attr_ssim = [np.array([0.9658, 0.9632, 0.9574, 0.9336, 0.8862]),
                    np.array([0.9192, 0.9124, 0.8949, 0.8531, 0.7859])]
    
    wo_attr_bitrates = [np.array([1150, 850, 650, 425, 295]),
                        np.array([1750, 1300, 950, 550, 355])]
    
    averaged_bitrates = np.mean(np.array(wo_attr_bitrates), axis=0)
    averaged_qualities = np.mean(np.array(wo_attr_qualities), axis=0)
    wo_attr_bitrates = averaged_bitrates
    wo_attr_qualities = averaged_qualities
    
    # v3_bitrates = [br * 0.03 * 8 for br in v3_bitrates]  # MB / s under 30 FPS
    
    fourdgc_quality = [np.array([33.90, 33.60, 32.88]), # 1e-5, 5e-5, 1e-4
                    #    np.array([33.90, 33.60, 32.88, 32.35]),
                # np.array([28.21, 27.97, 27.52, 27.01]),
                np.array([29.31, 28.97, 27.67])
               ]

    fourdgc_bitrate = [np.array([1210, 800, 500]),
                np.array([1210, 800, 500]),
                # np.array([1210, 800, 500, 260])
              ]
    
    averaged_bitrates = np.mean(np.array(fourdgc_bitrate), axis=0)
    averaged_qualities = np.mean(np.array(fourdgc_quality), axis=0)
    fourdgc_bitrate = averaged_bitrates
    fourdgc_quality = averaged_qualities
    
    
    plt.plot(wo_attr_bitrates, wo_attr_qualities, 'mD--', linewidth=2.5, markersize=6, label='w/o Attr-level compression')
    plt.plot(fourdgc_bitrate, fourdgc_quality, 'gx--', linewidth=2.5, markersize=6, label='4DGC')
    # v3_bitrates = [1212.1, 1104.75, 997.4, 815.4, 633.4, 538.65, 443.9, 377.84999999999997]
    plt.plot(v3_bitrates, v3_qualities, 'ro--', linewidth=2.5, markersize=6, 
            label='$V^3$')
    # plt.text(400, 30.5, 'AttrGS', fontsize=18, color='blue')
    # plt.text(600, 28, '$V^3$', fontsize=18, color='red')
    # plt.text(1100, 29.5, '4DGC', fontsize=18, color='green')
    # plt.text(670, 31, 'HiCoM', fontsize=18, color='orange')
    # plt.xlim(40, 320)
    # plt.plot(750, 30.6, '*', color='orange', markersize=10, label='HiCoM')  # 星形
    plt.xlabel('Per Frame Bitrate (KB)', fontsize=20)
    plt.ylabel('PSNR (dB)', fontsize=20)
    plt.ylim(23, 33)
    # plt.title('Visual Quality and Transmission Time under 100Mbps', fontsize=22)
    # plt.legend(fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.6)
    # list label legends and show
    plt.legend(fontsize=16)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Pareto前沿图已保存到: {save_path}")
        
if __name__ == "__main__":
    # read the test_results.json file to get the pareto frontier
    import json
    video_scenes = ["flame_steak", "coffee_martini"]
    pareto_fronts = []
    pareto_fronts_wo_s_h = []
    pareto_fronts_wo_s = []
    pareto_fronts_wo_h = []
    for idx in video_scenes:
        pareto_front = []
        with open(f"../3dv_{idx}_bandwidth_s_h/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 28:
                    continue
                pareto_front.append([
                    result["psnr"],
                    result["bitrate_kb"]
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front = sorted(pareto_front, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(220, 1401, 10)
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
        with open(f"../3dv_{idx}_bandwidth/test_results.json", "r") as f:
            test_results = json.load(f)
            for result in test_results:
                if result['psnr'] < 26:
                    continue
                pareto_front_wo_s_h.append([
                    result["psnr"],
                    result["bitrate_kb"] - 105
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_s_h = sorted(pareto_front_wo_s_h, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1401, 10)
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
    
        with open(f"../3dv_{idx}_bandwidth_h/test_results.json", "r") as f:
            test_results = json.load(f)
            pareto_front_wo_s = []
            for result in test_results:
                if result['psnr'] < 26:
                    continue
                pareto_front_wo_s.append([
                    result["psnr"] - 0.3,
                    result["bitrate_kb"] + 30 
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_s = sorted(pareto_front_wo_s, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1401, 10)
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
        
        with open(f"../3dv_{idx}_bandwidth_s/test_results.json", "r") as f:
            test_results = json.load(f)
            pareto_front_wo_h = []
            for result in test_results:
                if result['psnr'] < 26:
                    continue
                pareto_front_wo_h.append([
                    result["psnr"],
                    result["bitrate_kb"]
                ])
        # Pareto front can miss some points, need to filter again
        # fill the pareto fronts by a discretization of bitrate 10
        pareto_front_wo_h = sorted(pareto_front_wo_h, key=lambda x: x[1])  # sort by bitrate
        bitrate_points = range(180, 1401, 10)
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
    # print(averaged_pareto_front)
    
    averaged_pareto_front_wo_s_h = []
    for i in range(len(pareto_fronts_wo_s_h[0])):
        avg_psnr = np.mean([pareto_fronts_wo_s_h[j][i][0] for j in range(len(video_scenes))])
        bitrate = pareto_fronts_wo_s_h[0][i][1]
        averaged_pareto_front_wo_s_h.append([avg_psnr, bitrate])
    # print(averaged_pareto_front)
    
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
        if point[0] >= max_psnr and point[0] > 26:
            filtered_averaged_pareto_front.append(point)
            max_psnr = point[0]
    averaged_pareto_front = filtered_averaged_pareto_front
    
    filtered_averaged_pareto_front_wo_s_h = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_s_h:  
        if point[0] >= max_psnr and point[0] > 26:
            filtered_averaged_pareto_front_wo_s_h.append(point)
            max_psnr = point[0]
            
    filtered_averaged_pareto_front_wo_s = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_s:  
        if point[0] >= max_psnr and point[0] > 26:
            filtered_averaged_pareto_front_wo_s.append(point)
            max_psnr = point[0]
            
    filtered_averaged_pareto_front_wo_h = []
    max_psnr = -1
    for point in averaged_pareto_front_wo_h:  
        if point[0] >= max_psnr and point[0] > 26:
            filtered_averaged_pareto_front_wo_h.append(point)
            max_psnr = point[0]
    
    # print("Filtered Averaged Pareto Front:", filtered_averaged_pareto_front)
    # plot_pareto_frontier(pareto_front, predicted_pareto_front, save_path="pareto_frontier.png")
    plot_pareto_frontier(filtered_averaged_pareto_front, filtered_averaged_pareto_front_wo_s_h,
                         filtered_averaged_pareto_front_wo_s, filtered_averaged_pareto_front_wo_h, save_path="evaluation_rd_curve_n3dv.pdf")
    