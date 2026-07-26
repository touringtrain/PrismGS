import csv
import numpy as np
import json
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--results",
    nargs="+",
    default=[str(REPO_ROOT / "assets" / "example_scene" / "test_results.json")],
    help="One or more test_results.json files.",
)
parser.add_argument(
    "--trace",
    default=str(REPO_ROOT / "evaluation" / "Lumos5G-v1.0.csv"),
    help="Path to Lumos5G throughput trace.",
)
args = parser.parse_args()

def read_throughput_data(file_path):
    throughput_data_walk = np.array([])
    throughput_data_drive = np.array([])
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if row['mobility_mode'] == 'walking':
                throughput_data_walk = np.append(throughput_data_walk, float(row['Throughput']))
            elif row['mobility_mode'] == 'driving':
                throughput_data_drive = np.append(throughput_data_drive, float(row['Throughput']))
    return throughput_data_walk, throughput_data_drive

throughput_data_walk, throughput_data_drive = read_throughput_data(args.trace)
print(np.mean(throughput_data_walk))
print(np.mean(throughput_data_drive))
print(np.std(throughput_data_walk))
print(np.std(throughput_data_drive))


throughput_data_drive = throughput_data_drive / 8
throughput_data_walk = throughput_data_walk / 8

hicom_psnr = [np.array([32.14]), np.array([31.87]), np.array([31.48])]
hicom_ssim = [np.array([0.9797]), np.array([0.9668]), np.array([0.9755])]
hicom_bitrate = [np.array([710 * 30]), np.array([780 * 30]), np.array([750 * 30])]

unified_qp_psnr = [np.array([35.0814, 35.0556, 34.9598, 34.6683, 33.9556, 32.0219, 32.0219]),
                            np.array([34.1961, 34.1592, 34.0229, 33.6539, 32.8253, 30.7697, 26.7341]), 
                            np.array([36.2693, 36.2334, 36.0995, 35.7345, 34.7456, 32.3418, 27.4024])]
unified_qp_ssim = [np.array([0.9890, 0.9888, 0.9881, 0.9866, 0.9828, 0.9740, 0.9549]),
                            np.array([0.9833, 0.9832, 0.9821, 0.9796, 0.9738, 0.9591, 0.9269]),
                            np.array([0.9909, 0.9907, 0.9899, 0.9879, 0.9831, 0.9716, 0.9467])]
unified_bitrates = [np.array([1041.66, 1041.66, 816.66, 616.66, 466.66, 358.33, 358.33]),
                    np.array([1400, 1392, 1108.83, 850, 633.33, 483.33, 375]),
                    np.array([1212.1, 1104.75, 997.4, 815.4, 633.4, 538.65, 423.9])]

fourdgc_psnr = [np.array([32.66, 31.91, 30.41, 29.74]), # 1e-5, 3e-5, 5e-5, 1e-4
                np.array([32.66, 31.91, 30.41, 29.74]),
                np.array([32.66, 31.91, 30.41, 29.74])]

fourdgc_ssim = [np.array([0.9754, 0.9666, 0.9585, 0.9437]),
                np.array([0.9754, 0.9666, 0.9585, 0.9437]),
                np.array([0.9754, 0.9666, 0.9585, 0.9437])]

fourdgc_bitrate = [np.array([1000, 500, 416, 376]),
                np.array([1000, 500, 416, 376]),
                np.array([1000, 500, 416, 376])]

# unified_bitrates = unified_bitrates * 30

# 3e-4: 26.57
# 5e-5: 26.42

for i in range(len(unified_bitrates)):
    unified_bitrates[i] = unified_bitrates[i] * 30  # convert to kbps
    
for i in range(len(fourdgc_bitrate)):
    fourdgc_bitrate[i] = fourdgc_bitrate[i] * 30  # convert to kbps

# read the tested Pareto frontier

ours_psnrs = []
ours_ssims = []
ours_bitrates = []

for result_path in args.results:
    ours_psnr = np.array([])
    ours_bitrate = np.array([])
    ours_ssim = np.array([])
    with open(result_path, "r") as f:
        test_results = json.load(f)
        for result in test_results:
            if result["psnr"] < 28:
                continue
            ours_psnr = np.append(ours_psnr, result["psnr"])
            ours_ssim = np.append(ours_ssim, result["ssim"])
            ours_bitrate = np.append(ours_bitrate, result["bitrate_kb"])
    print("loaded results from", result_path)
        
    # reverse the arrays to have descending order of bitrate
    ours_psnr = ours_psnr[::-1]
    ours_ssim = ours_ssim[::-1]
    ours_bitrate = ours_bitrate[::-1] * 30  # convert to kbps
    ours_psnrs.append(ours_psnr)
    ours_ssims.append(ours_ssim)
    ours_bitrates.append(ours_bitrate)



def calculate_average_fps(throughput_data, psnrs, ssims, bitrates):
    fps_list = []
    psnr_list = []
    ssim_list = []
    for throughput in throughput_data:
        # 将吞吐量从Mbps转换为Kbps以便比较
        throughput_kbps = throughput * 1000
        
        # 找到能够支持的最高码率
        supported_bitrates = bitrates[bitrates <= throughput_kbps]
        
        if len(supported_bitrates) > 0:
            # 选择最高码率（第一个元素，因为数组是从高到低排序的）
            selected_bitrate = supported_bitrates[0]
            # 在30FPS下能够支持
            fps_list.append(30)
        else:
            # 如果没有码率能够支持，选择最低码率
            selected_bitrate = bitrates[-1]
            # 计算实际能够达到的FPS（按比例缩放）
            # 实际FPS = 30 * (吞吐量 / 所需码率)，但不能超过30
            actual_fps = min(30, 30 * (throughput_kbps / (selected_bitrate)))
            fps_list.append(actual_fps)
            
        psnr_list.append(psnrs[bitrates == selected_bitrate][0])
        ssim_list.append(ssims[bitrates == selected_bitrate][0])

    return fps_list, psnr_list, ssim_list

hicom_fps_walks, hicom_psnr_walks, hicom_ssim_walks = [], [], []
hicom_fps_drives, hicom_psnr_drives, hicom_ssim_drives = [], [], []
for i in range(len(hicom_psnr)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data_walk, hicom_psnr[i], hicom_ssim[i], hicom_bitrate[i])
    hicom_fps_walks.append(fps_walk)
    hicom_psnr_walks.append(psnr_walk)
    hicom_ssim_walks.append(ssim_walk)
    fps_drive, psnr_drive, ssim_drive = calculate_average_fps(throughput_data_drive, hicom_psnr[i], hicom_ssim[i], hicom_bitrate[i])
    hicom_fps_drives.append(fps_drive)
    hicom_psnr_drives.append(psnr_drive)
    hicom_ssim_drives.append(ssim_drive)
    print("Hicom: ", "fps walk:", np.mean(fps_walk), "fps_drive:", np.mean(fps_drive), "psnr_walk:", np.mean(psnr_walk), "psnr_drive:", np.mean(psnr_drive), "ssim_walk:", np.mean(ssim_walk), "ssim_drive:", np.mean(ssim_drive), "violation_walk:", np.mean(np.array(fps_walk) < 30), "violation_drive:", np.mean(np.array(fps_drive) < 30))
    
unified_fps_walks, unified_psnr_walks, unified_ssim_walks = [], [], []
unified_fps_drives, unified_psnr_drives, unified_ssim_drives = [], [], []
for i in range(len(unified_qp_psnr)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data_walk, unified_qp_psnr[i], unified_qp_ssim[i], unified_bitrates[i])
    unified_fps_walks.append(fps_walk)
    unified_psnr_walks.append(psnr_walk)
    unified_ssim_walks.append(ssim_walk)
    fps_drive, psnr_drive, ssim_drive = calculate_average_fps(throughput_data_drive, unified_qp_psnr[i], unified_qp_ssim[i], unified_bitrates[i])
    unified_fps_drives.append(fps_drive)
    unified_psnr_drives.append(psnr_drive)
    unified_ssim_drives.append(ssim_drive)

fourdgc_fps_walks, fourdgc_psnr_walks, fourdgc_ssim_walks = [], [], []
fourdgc_fps_drives, fourdgc_psnr_drives, fourdgc_ssim_drives = [], [], []
for i in range(len(fourdgc_psnr)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data_walk, fourdgc_psnr[i], fourdgc_ssim[i], fourdgc_bitrate[i])
    fourdgc_fps_walks.append(fps_walk)
    fourdgc_psnr_walks.append(psnr_walk)
    fourdgc_ssim_walks.append(ssim_walk)
    fps_drive, psnr_drive, ssim_drive = calculate_average_fps(throughput_data_drive, fourdgc_psnr[i], fourdgc_ssim[i], fourdgc_bitrate[i])
    fourdgc_fps_drives.append(fps_drive)
    fourdgc_psnr_drives.append(psnr_drive)
    fourdgc_ssim_drives.append(ssim_drive)


ours_fps_walks, ours_psnr_walks, ours_ssim_walks = [], [], []
ours_fps_drives, ours_psnr_drives, ours_ssim_drives = [], [], []
for i in range(len(ours_psnrs)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data_walk, ours_psnrs[i], ours_ssims[i], ours_bitrates[i])
    ours_fps_walks.append(fps_walk)
    ours_psnr_walks.append(psnr_walk)
    ours_ssim_walks.append(ssim_walk)
    fps_drive, psnr_drive, ssim_drive = calculate_average_fps(throughput_data_drive, ours_psnrs[i], ours_ssims[i], ours_bitrates[i])
    ours_fps_drives.append(fps_drive)
    ours_psnr_drives.append(psnr_drive)
    ours_ssim_drives.append(ssim_drive)

import matplotlib.pyplot as plt

# plot all obtained data: 
# plot for walking-smoothness, driving-smoothness, walking-quality, driving-quality


import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (12, 8)

# 计算统计数据
methods = ['HiCoM', '$V^3$', 'AdaGS']

# Walking 统计数据
walking_fpss, walking_violations, walking_psnrs, walking_ssims = [], [], [], []
for i in range(len(unified_qp_psnr)):
    hicom_fps_walk = hicom_fps_walks[i]
    hicom_psnr_walk = hicom_psnr_walks[i]
    hicom_ssim_walk = hicom_ssim_walks[i]
    
    unified_fps_walk = unified_fps_walks[i]
    unified_psnr_walk = unified_psnr_walks[i]
    unified_ssim_walk = unified_ssim_walks[i]
    
    fourdgc_fps_walk = fourdgc_fps_walks[i]
    fourdgc_psnr_walk = fourdgc_psnr_walks[i]
    fourdgc_ssim_walk = fourdgc_ssim_walks[i]
    
    our_fps_walk = ours_fps_walks[i]
    our_psnr_walk = ours_psnr_walks[i]
    our_ssim_walk = ours_ssim_walks[i]
    
    walking_fpss.append([np.mean(hicom_fps_walk), np.mean(unified_fps_walk), np.mean(fourdgc_fps_walk), np.mean(our_fps_walk)])
    walking_violations.append([np.mean(np.array(hicom_fps_walk) < 30), np.mean(np.array(unified_fps_walk) < 30), np.mean(np.array(fourdgc_fps_walk) < 30), np.mean(np.array(our_fps_walk) < 30)])
    walking_psnrs.append([np.mean(hicom_psnr_walk), np.mean(unified_psnr_walk), np.mean(fourdgc_psnr_walk), np.mean(our_psnr_walk)])
    walking_ssims.append([np.mean(hicom_ssim_walk), np.mean(unified_ssim_walk), np.mean(fourdgc_ssim_walk), np.mean(our_ssim_walk)])

# Driving 统计数据
driving_fpss, driving_violations, driving_psnrs, driving_ssims = [], [], [], []
for i in range(len(unified_qp_psnr)):
    hicom_fps_drive = hicom_fps_drives[i]
    hicom_psnr_drive = hicom_psnr_drives[i]
    hicom_ssim_drive = hicom_ssim_drives[i]
    
    unified_fps_drive = unified_fps_drives[i]
    unified_psnr_drive = unified_psnr_drives[i]
    unified_ssim_drive = unified_ssim_drives[i]
    
    fourdgc_fps_drive = fourdgc_fps_drives[i]
    fourdgc_psnr_drive = fourdgc_psnr_drives[i]
    fourdgc_ssim_drive = fourdgc_ssim_drives[i]
    
    our_fps_drive = ours_fps_drives[i]
    our_psnr_drive = ours_psnr_drives[i]
    our_ssim_drive = ours_ssim_drives[i]
    
    driving_fpss.append([np.mean(hicom_fps_drive), np.mean(unified_fps_drive), np.mean(fourdgc_fps_drive), np.mean(our_fps_drive)])
    driving_violations.append([np.mean(np.array(hicom_fps_drive) < 30), np.mean(np.array(unified_fps_drive) < 30), np.mean(np.array(fourdgc_fps_drive) < 30), np.mean(np.array(our_fps_drive) < 30)])
    driving_psnrs.append([np.mean(hicom_psnr_drive), np.mean(unified_psnr_drive), np.mean(fourdgc_psnr_drive), np.mean(our_psnr_drive)])
    driving_ssims.append([np.mean(hicom_ssim_drive), np.mean(unified_ssim_drive), np.mean(fourdgc_ssim_drive), np.mean(our_ssim_drive)])

# # 1. Walking - Smoothness
# fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))

# # 平均FPS
# patterns = ['/', '\\', 'x']
# # add different hatches for each bars
# bars1 = ax1.bar(methods, np.mean(walking_fpss, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars1, patterns):
#     bar.set_hatch(pattern)

# ax1.set_ylabel('Average FPS', fontsize=18)
# # ax1.set_title('Average FPS', fontsize=18)
# ax1.set_ylim(20, 30)
# # 设置x轴和y轴标签字号
# ax1.tick_params(axis='x', labelsize=18)
# ax1.tick_params(axis='y', labelsize=18)
# # add error bars
# ax1.errorbar(methods, np.mean(walking_fpss, axis=0), yerr=np.std(walking_fpss, axis=0), fmt='none', ecolor='black', capsize=5)

# # FPS Violation Rate
# bars2 = ax2.bar(methods, np.mean(walking_violations, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars2, patterns):
#     bar.set_hatch(pattern)

# ax2.set_ylabel('Violation Rate', fontsize=18)
# # ax2.set_title('FPS Violation Rate', fontsize=18)
# ax2.set_ylim(0, 0.4)
# # 设置x轴和y轴标签字号
# ax2.tick_params(axis='x', labelsize=18)
# ax2.tick_params(axis='y', labelsize=18)
# ax2.errorbar(methods, np.mean(walking_violations, axis=0), yerr=np.std(walking_violations, axis=0), fmt='none', ecolor='black', capsize=5)

# fig1.tight_layout()
# plt.savefig('evaluation_walking_smoothness.pdf', dpi=300, bbox_inches='tight')
# plt.close(fig1)
# print("Saved: evaluation_walking_smoothness.pdf")

# # 2. Walking - Quality
# fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))

# # PSNR
# bars1 = ax1.bar(methods, np.mean(walking_psnrs, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars1, patterns):
#     bar.set_hatch(pattern)

# ax1.set_ylabel('PSNR (dB)', fontsize=18)
# ax1.set_ylim(20, 36)
# # ax1.set_title('Average PSNR', fontsize=18)
# # 设置x轴和y轴标签字号
# ax1.tick_params(axis='x', labelsize=18)
# ax1.tick_params(axis='y', labelsize=18)
# ax1.errorbar(methods, np.mean(walking_psnrs, axis=0), yerr=np.std(walking_psnrs, axis=0), fmt='none', ecolor='black', capsize=5)

# # SSIM
# bars2 = ax2.bar(methods, np.mean(walking_ssims, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars2, patterns):
#     bar.set_hatch(pattern)
# ax2.set_ylabel('SSIM', fontsize=18)
# # ax2.set_title('Average SSIM', fontsize=18)
# ax2.set_ylim(0.9, 1.0)
# # 设置x轴和y轴标签字号
# ax2.tick_params(axis='x', labelsize=18)
# ax2.tick_params(axis='y', labelsize=18)
# ax2.errorbar(methods, np.mean(walking_ssims, axis=0), yerr=np.std(walking_ssims, axis=0), fmt='none', ecolor='black', capsize=5)

# fig2.tight_layout()
# plt.savefig('evaluation_walking_quality.pdf', dpi=300, bbox_inches='tight')
# plt.close(fig2)
# print("Saved: evaluation_walking_quality.pdf")

# # 3. Driving - Smoothness
# fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))

# # 平均FPS
# bars1 = ax1.bar(methods, np.mean(driving_fpss, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars1, patterns):
#     bar.set_hatch(pattern)

# ax1.set_ylabel('Average FPS', fontsize=18)
# # ax1.set_title('Average FPS', fontsize=18)
# ax1.set_ylim(20, 30)
# # 设置x轴和y轴标签字号
# ax1.tick_params(axis='x', labelsize=18)
# ax1.tick_params(axis='y', labelsize=18)
# # add error bars
# ax1.errorbar(methods, np.mean(driving_fpss, axis=0), yerr=np.std(driving_fpss, axis=0), fmt='none', ecolor='black', capsize=5)


# # FPS Violation Rate
# bars2 = ax2.bar(methods, np.mean(driving_violations, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars2, patterns):
#     bar.set_hatch(pattern)

# ax2.set_ylabel('Violation Rate', fontsize=18)
# # ax2.set_title('FPS Violation Rate', fontsize=18)
# ax2.set_ylim(0, 0.65)
# # 设置x轴和y轴标签字号
# ax2.tick_params(axis='x', labelsize=18)
# ax2.tick_params(axis='y', labelsize=18)
# ax2.errorbar(methods, np.mean(driving_violations, axis=0), yerr=np.std(driving_violations, axis=0), fmt='none', ecolor='black', capsize=5)

# fig3.tight_layout()
# plt.savefig('evaluation_driving_smoothness.pdf', dpi=300, bbox_inches='tight')
# plt.close(fig3)
# print("Saved: evaluation_driving_smoothness.pdf")

# # 4. Driving - Quality
# fig4, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 5))

# # PSNR
# bars1 = ax1.bar(methods, np.mean(driving_psnrs, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars1, patterns):
#     bar.set_hatch(pattern)
# ax1.set_ylabel('PSNR (dB)', fontsize=18)
# ax1.set_ylim(20, 36)
# # ax1.set_title('Average PSNR', fontsize=18)
# # 设置x轴和y轴标签字号
# ax1.tick_params(axis='x', labelsize=18)
# ax1.tick_params(axis='y', labelsize=18)
# ax1.errorbar(methods, np.mean(driving_psnrs, axis=0), yerr=np.std(driving_psnrs, axis=0), fmt='none', ecolor='black', capsize=5)

# # SSIM
# bars2 = ax2.bar(methods, np.mean(driving_ssims, axis=0), color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1.2)
# for bar, pattern in zip(bars2, patterns):
#     bar.set_hatch(pattern)

# ax2.set_ylabel('SSIM', fontsize=18)
# # ax2.set_title('Average SSIM', fontsize=18)
# ax2.set_ylim(0.9, 1.0)
# # 设置x轴和y轴标签字号
# ax2.tick_params(axis='x', labelsize=18)
# ax2.tick_params(axis='y', labelsize=18)
# ax2.errorbar(methods, np.mean(driving_ssims, axis=0), yerr=np.std(driving_ssims, axis=0), fmt='none', ecolor='black', capsize=5)

# fig4.tight_layout()
# plt.savefig('evaluation_driving_quality.pdf', dpi=300, bbox_inches='tight')
# plt.close(fig4)
# print("Saved: evaluation_driving_quality.pdf")
# print(walking_psnrs)
# 打印详细数据

HiCoM_data = []
V3_data = []
AdaGS_data = []

print("\n=== Walking Scenario ===")
print("HiCoM: FPS =", np.mean(walking_fpss, axis=0)[0], ", Violation Rate =", np.mean(walking_violations, axis=0)[0], ", PSNR =", np.mean(walking_psnrs, axis=0)[0], ", SSIM =", np.mean(walking_ssims, axis=0)[0])
print("$V^3$: FPS =", np.mean(walking_fpss, axis=0)[1], ", Violation Rate =", np.mean(walking_violations, axis=0)[1], ", PSNR =", np.mean(walking_psnrs, axis=0)[1], ", SSIM =", np.mean(walking_ssims, axis=0)[1])
print("4DGC: FPS =", np.mean(walking_fpss, axis=0)[2], ", Violation Rate =", np.mean(walking_violations, axis=0)[2], ", PSNR =", np.mean(walking_psnrs, axis=0)[2], ", SSIM =", np.mean(walking_ssims, axis=0)[2])
print("AdaGS: FPS =", np.mean(walking_fpss, axis=0)[3], ", Violation Rate =", np.mean(walking_violations, axis=0)[3], ", PSNR =", np.mean(walking_psnrs, axis=0)[3], ", SSIM =", np.mean(walking_ssims, axis=0)[3])

# the average of FPS deviation in each scene
# print("FPS deviation (HiCoM):", np.mean(np.std(hicom_fps_walks, axis=1)))
# print("FPS deviation (V3):", np.mean(np.std(unified_fps_walks, axis=1)))
# print("FPS deviation (AdaGS):", np.mean(np.std(ours_fps_walks, axis=1)))

# the average of PSNR deviation in each scene
# print("PSNR deviation (V3):", np.mean(np.std(unified_psnr_walks, axis=1)))
# print("PSNR deviation (AdaGS):", np.mean(np.std(ours_psnr_walks, axis=1)))

print("\n=== Driving Scenario ===")
print("HiCoM: FPS =", np.mean(driving_fpss, axis=0)[0], ", Violation Rate =", np.mean(driving_violations, axis=0)[0], ", PSNR =", np.mean(driving_psnrs, axis=0)[0], ", SSIM =", np.mean(driving_ssims, axis=0)[0])
print("$V^3$: FPS =", np.mean(driving_fpss, axis=0)[1], ", Violation Rate =", np.mean(driving_violations, axis=0)[1], ", PSNR =", np.mean(driving_psnrs, axis=0)[1], ", SSIM =", np.mean(driving_ssims, axis=0)[1])
print("4DGC: FPS =", np.mean(driving_fpss, axis=0)[2], ", Violation Rate =", np.mean(driving_violations, axis=0)[2], ", PSNR =", np.mean(driving_psnrs, axis=0)[2], ", SSIM =", np.mean(driving_ssims, axis=0)[2])
print("AdaGS: FPS =", np.mean(driving_fpss, axis=0)[3], ", Violation Rate =", np.mean(driving_violations, axis=0)[3], ", PSNR =", np.mean(driving_psnrs, axis=0)[3], ", SSIM =", np.mean(driving_ssims, axis=0)[3])

# the average of FPS deviation in each scene
print("FPS deviation (HiCoM):", np.mean(np.std(hicom_fps_drives, axis=1)))
print("FPS deviation (V3):", np.mean(np.std(unified_fps_drives, axis=1)))
print("FPS deviation (4DGC):", np.mean(np.std(fourdgc_fps_drives, axis=1)))
print("FPS deviation (AdaGS):", np.mean(np.std(ours_fps_drives, axis=1)))

# the average of PSNR deviation in each scene
print("PSNR deviation (HiCoM):", np.mean(np.std(hicom_psnr_drives, axis=1)))
print("PSNR deviation (V3):", np.mean(np.std(unified_psnr_drives, axis=1)))
print("PSNR deviation (4DGC):", np.mean(np.std(fourdgc_psnr_drives, axis=1)))
print("PSNR deviation (AdaGS):", np.mean(np.std(ours_psnr_drives, axis=1)))
