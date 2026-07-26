import numpy as np
import matplotlib.pyplot as plt
import json

throughput_data = np.arange(0, 600, 10)

# fps_static = [] 

# for i in range(len(throughput_data)):
#     fps_static.append(min(throughput_data[i] / 8 / 0.75, 30))
        
# psnr_static = []
# for i in range(len(throughput_data)):
#     psnr_static.append(30.65)
    
# ssim_static = []
# for i in range(len(throughput_data)):
#     ssim_static.append(0.9654)

# # print(fps_static)

# # using a naive bandwidth-adaptive method

# psnr_base = [36.27, 36.23, 36.09, 35.73, 34.74, 32.34, 27.90]
# ssim_base = [0.9909, 0.9907, 0.9899, 0.9879, 0.9831, 0.9716, 0.9467]
# size_coefficient = [240 / 156, 240 / 156, 196 / 156, 1, 128 / 156, 108 / 156, 92 / 156]

# def adaptive(throughput_data, psnr, ssim, size_coefficient):
#     fps_adaptive = []
#     psnr_adaptive = []
#     ssim_adaptive = []
#     for i in range(len(throughput_data)):
#         max_fps = (throughput_data[i] / 8) / size_coefficient[-1]
#         fps_adaptive.append(min(max_fps, 30))
#         if max_fps < 30:
#             psnr_adaptive.append(psnr[-1])
#             ssim_adaptive.append(ssim[-1])
#         else:
#             for j in range(len(size_coefficient)):
#                 if throughput_data[i] / 8 >= 30 * size_coefficient[j]:
#                     # print(max_fps)
#                     psnr_adaptive.append(psnr[j])
#                     ssim_adaptive.append(ssim[j])
#                     break
#     return fps_adaptive, psnr_adaptive, ssim_adaptive

# fps_adaptive, psnr_adaptive, ssim_adaptive = adaptive(throughput_data, psnr_base, ssim_base, size_coefficient)

# import json
# import numpy as np

# ours_psnr = np.array([])
# ours_bitrate = np.array([])
# ours_ssim = np.array([])

# with open("../quality_testset/test_results.json", "r") as f:
#     test_results = json.load(f)
#     for result in test_results:
#         if result["bitrate_kb"] < 300:
#             continue
#         ours_psnr = np.append(ours_psnr, result["psnr"])
#         ours_ssim = np.append(ours_ssim, result["ssim"])
#         ours_bitrate = np.append(ours_bitrate, result["bitrate_kb"])
        
# # reverse the arrays to have descending order of bitrate
# ours_psnr = ours_psnr[::-1]
# ours_ssim = ours_ssim[::-1]
# ours_bitrate = ours_bitrate[::-1] * 30  # convert to kbps
# print(ours_bitrate)

# def calculate_average_fps(throughput_data, psnrs, ssims, bitrates):
#     fps_list = []
#     psnr_list = []
#     ssim_list = []
#     for throughput in throughput_data:
#         # 将吞吐量从Mbps转换为Kbps以便比较
#         throughput_kbps = throughput * 1000 / 8
        
#         # 找到能够支持的最高码率
#         supported_bitrates = bitrates[bitrates <= throughput_kbps]
        
#         if len(supported_bitrates) > 0:
#             # 选择最高码率（第一个元素，因为数组是从高到低排序的）
#             selected_bitrate = supported_bitrates[0]
#             # 在30FPS下能够支持
#             fps_list.append(30)
#         else:
#             # 如果没有码率能够支持，选择最低码率
#             selected_bitrate = bitrates[-1]
#             # 计算实际能够达到的FPS（按比例缩放）
#             # 实际FPS = 30 * (吞吐量 / 所需码率)，但不能超过30
#             actual_fps = min(30, 30 * (throughput_kbps / (selected_bitrate)))
#             fps_list.append(actual_fps)
            
#         psnr_list.append(psnrs[bitrates == selected_bitrate][0])
#         ssim_list.append(ssims[bitrates == selected_bitrate][0])

#     return fps_list, psnr_list, ssim_list


# our_fps, our_psnr, our_ssim = calculate_average_fps(throughput_data, ours_psnr, ours_ssim, ours_bitrate)

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
# unified_bitrates = unified_bitrates * 30

for i in range(len(unified_bitrates)):
    unified_bitrates[i] = unified_bitrates[i] * 30  # convert to kbps

# read the tested Pareto frontier

ours_psnrs = []
ours_ssims = []
ours_bitrates = []

scene_idx = [1, 4, 5]
for idx in scene_idx:
    ours_psnr = np.array([])
    ours_bitrate = np.array([])
    ours_ssim = np.array([])
    with open(f"../4k_{idx}_bandwidth/test_results.json", "r") as f:
        test_results = json.load(f)
        for result in test_results:
            if result["psnr"] < 28:
                continue
            ours_psnr = np.append(ours_psnr, result["psnr"])
            ours_ssim = np.append(ours_ssim, result["ssim"])
            ours_bitrate = np.append(ours_bitrate, result["bitrate_kb"])
    print("done scene", idx)
        
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

test_hicom_fps, test_hicom_psnr, test_hicom_ssim = [], [], []
for i in range(len(hicom_psnr)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data / 8, hicom_psnr[i], hicom_ssim[i], hicom_bitrate[i])
    test_hicom_fps.append(fps_walk)
    test_hicom_psnr.append(psnr_walk)
    test_hicom_ssim.append(ssim_walk)

    
test_unified_fps, test_unified_psnr, test_unified_ssim = [], [], []
for i in range(len(unified_qp_psnr)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data / 8, unified_qp_psnr[i], unified_qp_ssim[i], unified_bitrates[i])
    test_unified_fps.append(fps_walk)
    test_unified_psnr.append(psnr_walk)
    test_unified_ssim.append(ssim_walk)

test_ours_fps, test_ours_psnr, test_ours_ssim = [], [], []
for i in range(len(ours_psnrs)):
    fps_walk, psnr_walk, ssim_walk = calculate_average_fps(throughput_data / 8, ours_psnrs[i], ours_ssims[i], ours_bitrates[i])
    test_ours_fps.append(fps_walk)
    test_ours_psnr.append(psnr_walk)
    test_ours_ssim.append(ssim_walk)
    
line_width = 3
line_style = ['-', '--', '-.']
marker = ['o', 's', '^']
markersize = 7
markevery = 5


plt.figure(figsize=(8, 6))
plt.plot(throughput_data, np.mean(test_hicom_fps, axis=0), label='HiCoM', color='blue', linestyle=line_style[0], linewidth=line_width, marker='o', markersize=markersize, markevery=2)
plt.plot(throughput_data, np.mean(test_unified_fps, axis=0), label='$V^3$', color='orange', linestyle=line_style[1], linewidth=line_width, marker='s', markersize=markersize, markevery=2)
plt.plot(throughput_data, np.mean(test_ours_fps, axis=0), label='AdaGS', color='green', linestyle=line_style[2], linewidth=line_width, marker='^', markersize=markersize, markevery=2)
plt.xlabel('Bandwidth (Mbps)', fontsize=22)
plt.ylabel('FPS at Client', fontsize=22)
# plt.title('FPS vs Bandwidth', fontsize=22)
plt.xlim(-10, 320)
plt.ylim(0, 35)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=22, loc='lower right')

# for each method, draw a vertical dashed line when fps reaches 30
for i in range(len(throughput_data)):
    if np.mean(test_hicom_fps, axis=0)[i] >= 30:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=30, color='blue', linestyle='--', alpha=0.5, linewidth=3)
        # add text for bandwidth , put the text near x axis
        plt.text(throughput_data[i] + 5, 0 + 0.2 * (35 - 0), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='blue', rotation=90, va='center', alpha=1)
        break
for i in range(len(throughput_data)):
    if np.mean(test_unified_fps, axis=0)[i] >= 30:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=30, color='orange', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 0 + 0.2 * (35 - 0), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='orange', rotation=90, va='center', alpha=1)
        break
for i in range(len(throughput_data)):
    if np.mean(test_ours_fps, axis=0)[i] >= 30:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=30, color='green', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 0 + 0.2 * (35 - 0), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='green', rotation=90, va='center', alpha=1)
        break


plt.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig('evaluation_sensitivity_fps.pdf', dpi=300)




plt.figure(figsize=(8, 6))
plt.plot(throughput_data, np.mean(test_hicom_psnr, axis=0), label='HiCoM', color='blue', linestyle=line_style[0], linewidth=line_width, marker='o', markersize=markersize, markevery=markevery)
plt.plot(throughput_data, np.mean(test_unified_psnr, axis=0), label='$V^3$', color='orange', linestyle=line_style[1], linewidth=line_width, marker='s', markersize=markersize, markevery=markevery)
plt.plot(throughput_data, np.mean(test_ours_psnr, axis=0), label='AdaGS', color='green', linestyle=line_style[2], linewidth=line_width, marker='^', markersize=markersize, markevery=markevery)
plt.xlabel('Bandwidth (Mbps)', fontsize=22)
plt.ylabel('PSNR (dB)', fontsize=22)
# plt.title('PSNR vs Bandwidth', fontsize=22)
plt.xlim(-10, 600)
plt.ylim(25, 37)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=22, loc='lower right')

for i in range(len(throughput_data)):
    if np.mean(test_unified_psnr, axis=0)[i] == np.mean(test_unified_psnr, axis=0)[-1]:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=np.mean(test_unified_psnr, axis=0)[i], color='orange', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 25 + 0.2 * (38 - 25), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='orange', rotation=90, va='center', alpha=1)
        break     
for i in range(len(throughput_data)):
    if np.mean(test_ours_psnr, axis=0)[i] >= np.mean(test_ours_psnr, axis=0)[-1] - 0.05:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=np.mean(test_ours_psnr, axis=0)[i], color='green', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 25 + 0.2 * (38 - 25), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='green', rotation=90, va='center', alpha=1)
        break
    
plt.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig('evaluation_sensitivity_psnr.pdf', dpi=300)

plt.figure(figsize=(8, 6))
plt.plot(throughput_data, np.mean(test_hicom_ssim, axis=0), label='HiCoM', color='blue', linestyle=line_style[0], linewidth=line_width, marker='o', markersize=markersize, markevery=markevery)
plt.plot(throughput_data, np.mean(test_unified_ssim, axis=0), label='$V^3$', color='orange', linestyle=line_style[1], linewidth=line_width, marker='s', markersize=markersize, markevery=markevery)
plt.plot(throughput_data, np.mean(test_ours_ssim, axis=0), label='AdaGS', color='green', linestyle=line_style[2], linewidth=line_width, marker='^', markersize=markersize, markevery=markevery)
plt.xlabel('Bandwidth (Mbps)', fontsize=22)
plt.ylabel('SSIM', fontsize=22)
# plt.title('SSIM vs Bandwidth', fontsize=22)
plt.xlim(-10, 600)
plt.ylim(0.92, 1.0)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=22, loc='lower right')

for i in range(len(throughput_data)):
    if np.mean(test_unified_ssim, axis=0)[i] == np.mean(test_unified_ssim, axis=0)[-1]:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=np.mean(test_unified_ssim, axis=0)[i], color='orange', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 0.92 + 0.2 * (1.0 - 0.92), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='orange', rotation=90, va='center', alpha=1)
        break
for i in range(len(throughput_data)):
    if np.mean(test_ours_ssim, axis=0)[i] >= np.mean(test_ours_ssim, axis=0)[-1] - 0.0005:
        plt.vlines(x=throughput_data[i], ymin=0, ymax=np.mean(test_ours_ssim, axis=0)[i], color='green', linestyle='--', alpha=0.5, linewidth=3)
        plt.text(throughput_data[i] + 5, 0.92 + 0.2 * (1.0 - 0.92), f'{throughput_data[i]:.0f} Mbps', fontsize=20, color='green', rotation=90, va='center', alpha=1)
        break

plt.tick_params(axis='both', which='major', labelsize=22)
plt.tight_layout()
plt.savefig('evaluation_sensitivity_ssim.pdf', dpi=300) 
