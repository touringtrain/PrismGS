import csv

# read the first 200 element of "Throughput" column from the CSV file
def read_throughput_data(file_path):
    throughput_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < 416:
                continue
            if i >= 466:
                break
            throughput_data.append(float(row['Throughput']))
    return throughput_data

throughput_data = read_throughput_data('../motivation_study/Lumos5G-v1.0.csv')

# minus 20% to simulate real-world network condition
throughput_data = [x / 2.5 for x in throughput_data]
# print(throughput_data)

def read_psnr_data(file_path):
    psnr_data = []
    # find the line with the format: "PSNR: 00.00 dB" and extract the PSNR value
    with open(file_path, 'r') as f:
        for line in f:
            if "PSNR:" in line:
                parts = line.split()
                for part in parts:
                    if part.replace('.', '', 1).isdigit():
                        psnr_data.append(float(part))
                        break
            if len(psnr_data) >= 50:
                break
    return psnr_data

psnr_data = read_psnr_data('../../videogs/VideoGS/output_log/eval_5_umbrella_prune_60.txt')

# print(psnr_data)

# using non-bandwidth-adaptive method

print("Average Throughput: ", sum(throughput_data) / len(throughput_data), "Variance: ", sum((x - sum(throughput_data) / len(throughput_data)) ** 2 for x in throughput_data) / len(throughput_data))

psnr_static = []
fps_static = [] 

for i in range(len(throughput_data)):
    fps_static.append(min(throughput_data[i] / 8 / 0.78, 30))
    psnr_static.append(30.65)  # always use the 4th config
    
    
# print(fps_static)

# using a naive bandwidth-adaptive method

quality_coefficient = [36.27 / 35.73, 36.23 / 35.73, 36.09 / 35.73, 1, 34.74 / 35.73, 32.34 / 35.73, 27.90 / 35.73]

size_coefficient = [240 / 156, 240 / 156, 196 / 156, 1, 128 / 156, 108 / 156, 92 / 156]

def adaptive(throughput_data, psnr_data, quality_coefficient, size_coefficient):
    fps_adaptive = []
    psnr_adaptive = []
    for i in range(len(throughput_data)):
        max_fps = (throughput_data[i] / 8) / size_coefficient[-1]
        fps_adaptive.append(min(max_fps, 30))
        
        if max_fps < 30:
            psnr_adaptive.append(psnr_data[1] * quality_coefficient[-1])
        else:
            for j in range(len(size_coefficient)):
                if max_fps >= 30 / size_coefficient[j]:
                    # print(max_fps)
                    psnr_adaptive.append(psnr_data[1] * quality_coefficient[j])
                    break
    return fps_adaptive, psnr_adaptive

fps_adaptive, psnr_adaptive = adaptive(throughput_data, psnr_data, quality_coefficient, size_coefficient)

fourdgc_quality = [32.66, 31.91, 30.41, 28.74]

fourdgc_bitrate = [1000, 500, 416, 376]

def fourdgc_adaptive(throughput_data, fourdgc_quality, fourdgc_bitrate):
    fps_adaptive = []
    psnr_adaptive = []
    for i in range(len(throughput_data)):
        max_fps = (throughput_data[i] / 8) / fourdgc_bitrate[-1] * 1000
        fps_adaptive.append(min(max_fps, 30))
        
        if max_fps < 30:
            psnr_adaptive.append(fourdgc_quality[-1])
        else:
            for j in range(len(fourdgc_bitrate)):
                if max_fps >= 30 / fourdgc_bitrate[j]:
                    # print(max_fps)
                    psnr_adaptive.append(fourdgc_quality[j])
                    break
    return fps_adaptive, psnr_adaptive

fourdgc_fps_adaptive, fourdgc_psnr_adaptive = fourdgc_adaptive(throughput_data, fourdgc_quality, fourdgc_bitrate)

# print(fps_adaptive)
# print(psnr_adaptive)

# print("Average FPS (Static Method): ", sum(fps_static) / len(fps_static), "Variance: ", sum((x - sum(fps_static) / len(fps_static)) ** 2 for x in fps_static) / len(fps_static))
# print("Average FPS (Adaptive Method): ", sum(fps_adaptive) / len(fps_adaptive), "Variance: ", sum((x - sum(fps_adaptive) / len(fps_adaptive)) ** 2 for x in fps_adaptive) / len(fps_adaptive))

# print("Average PSNR (Static Method): ", sum(psnr_data) / len(psnr_data), "Variance: ", sum((x - sum(psnr_data) / len(psnr_data)) ** 2 for x in psnr_data) / len(psnr_data))
# print("Average PSNR (Adaptive Method): ", sum(psnr_adaptive) / len(psnr_adaptive), "Variance: ", sum((x - sum(psnr_adaptive) / len(psnr_adaptive)) ** 2 for x in psnr_adaptive) / len(psnr_adaptive))

# plot the fps and psnr data separately

import json
import numpy as np

ours_psnr = np.array([])
ours_bitrate = np.array([])
ours_ssim = np.array([])

ours_position_low = []
ours_color = []
ours_opacity = []
ours_scale = []
ours_rotation = []

with open("../4k_5_bandwidth/test_results.json", "r") as f:
    test_results = json.load(f)
    for result in test_results:
        if result["psnr"] < 28:
            continue
        ours_psnr = np.append(ours_psnr, result["psnr"])
        ours_ssim = np.append(ours_ssim, result["ssim"])
        ours_bitrate = np.append(ours_bitrate, result["bitrate_kb"])
        ours_position_low.append(result["position_low"])
        ours_color.append(result["color"])
        ours_opacity.append(result["opacity"])
        ours_scale.append(result["scale"])
        ours_rotation.append(result["rotation"])
        
# reverse the arrays to have descending order of bitrate
ours_psnr = ours_psnr[::-1]
ours_ssim = ours_ssim[::-1]
ours_bitrate = ours_bitrate[::-1] * 30  # convert to kbps
# print(ours_bitrate)
ours_position_low = ours_position_low[::-1]
ours_color = ours_color[::-1]
ours_opacity = ours_opacity[::-1]
ours_scale = ours_scale[::-1]
ours_rotation = ours_rotation[::-1]

def calculate_average_fps(throughput_data, psnrs, ssims, bitrates, positions, colors, opacities, scales, rotations):
    fps_list = []
    psnr_list = []
    ssim_list = []
    ours_position_low = []
    ours_color = []
    ours_opacity = []
    ours_scale = []
    ours_rotation = []
    for throughput in throughput_data:
        # 将吞吐量从Mbps转换为Kbps以便比较
        throughput_kbps = throughput * 1000 / 8
        
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
        ours_position_low.append(positions[bitrates.tolist().index(selected_bitrate)])
        ours_color.append(colors[bitrates.tolist().index(selected_bitrate)])
        ours_opacity.append(opacities[bitrates.tolist().index(selected_bitrate)])
        ours_scale.append(scales[bitrates.tolist().index(selected_bitrate)])
        ours_rotation.append(rotations[bitrates.tolist().index(selected_bitrate)])
    return fps_list, psnr_list, ssim_list, ours_position_low, ours_color, ours_opacity, ours_scale, ours_rotation


our_fps, our_psnr, our_ssim, ours_position_low, ours_color, ours_opacity, ours_scale, ours_rotation = \
    calculate_average_fps(throughput_data, ours_psnr, ours_ssim, ours_bitrate, ours_position_low, ours_color, ours_opacity, ours_scale, ours_rotation)

import matplotlib.pyplot as plt

# 创建包含两个共享x轴的子图的图表
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [1, 1.5, 1.5, 1.5]})

# 上半张图：吞吐量折线图
ax1.plot(throughput_data[:15], label='Throughput', color='black', linestyle='-', linewidth=2)
ax1.set_ylabel('Bandwidth\n(Mbps)', fontsize=18)
ax1.set_ylim(0, 300)
ax1.tick_params(axis='y', labelsize=18)
ax1.grid(True, alpha=0.3, linestyle='--')

line_style  = ['--', '-.', '-']
mark_style = ['o', 's', '^']

# 第二张图：FPS对比折线图
ax2.plot(fourdgc_fps_adaptive[:15], label='4DGC', color='blue', linestyle=line_style[0], marker=mark_style[0], linewidth=2)
ax2.plot(fps_adaptive[:15], label='$V^3$', color='orange', linestyle=line_style[1], marker=mark_style[1], linewidth=2)
ax2.plot(our_fps[:15], label='AdaGS', color='green', linestyle=line_style[2], marker=mark_style[2], linewidth=2)
ax2.set_ylabel('Client FPS', fontsize=18)
ax2.set_ylim(0, 35)
ax2.tick_params(axis='y', labelsize=18)
ax2.grid(True, alpha=0.3, linestyle='--')

# 第三张图：PSNR对比折线图
ax3.plot(fourdgc_psnr_adaptive[:15], label='4DGC', color='blue', linestyle=line_style[0], marker=mark_style[0], linewidth=2)
ax3.plot(psnr_adaptive[:15], label='$V^3$', color='orange', linestyle=line_style[1], marker=mark_style[1], linewidth=2)
ax3.plot(our_psnr[:15], label='AdaGS', color='green', linestyle=line_style[2], marker=mark_style[2], linewidth=2)
ax3.set_ylabel('PSNR (dB)', fontsize=18)
ax3.set_ylim(20, 38)
ax3.tick_params(axis='y', labelsize=18)
ax3.grid(True, alpha=0.3, linestyle='--')

# 第四张图：属性QP配置
ax4.plot(ours_color[:15], label='Color', color='purple', linestyle=line_style[1], linewidth=2, marker=mark_style[1])
ax4.plot(ours_opacity[:15], label='Opacity', color='grey', linestyle=line_style[2], linewidth=2, marker=mark_style[2])
ax4.plot(ours_position_low[:15], label='Position_low', color='red', linestyle=line_style[0], linewidth=2, marker=mark_style[0])
# ax4.plot(ours_scale[:15], label='Scale', color='orange', linestyle='-', linewidth=2, marker='D', markersize=3)
# ax4.plot(ours_rotation[:15], label='Rotation', color='purple', linestyle=line_style[2], linewidth=2, marker=mark_style[2])

# 在x=2.5-4.5, 6.5-10.5, 11.5-13.5间添加半透明阴影，并且添加到图例中
highlight1 = ax4.axvspan(2.5, 4.5, color='yellow', alpha=0.3, label='Operating Region')
highlight2 = ax4.axvspan(6.5, 10.5, color='yellow', alpha=0.3)
highlight3 = ax4.axvspan(11.5, 13.5, color='yellow', alpha=0.3)


ax4.set_xlabel('Time (s)', fontsize=18)
ax4.set_ylabel('Attribute QP', fontsize=18)
ax4.tick_params(axis='y', labelsize=18)
ax4.tick_params(axis='x', labelsize=18)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.invert_yaxis()
ax4.set_ylim(55, -5)

# 为每个子图添加标签 (a), (b), (c), (d)
ax1.text(0.02, 0.90, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
ax2.text(0.02, 0.90, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
ax3.text(0.02, 0.90, '(c)', transform=ax3.transAxes, fontsize=16, fontweight='bold', va='top')
ax4.text(0.02, 0.90, '(d)', transform=ax4.transAxes, fontsize=16, fontweight='bold', va='top')

# 方法对比图例（放在第三张图下方）
method_handles, method_labels = ax2.get_legend_handles_labels()

fig.legend(method_handles, method_labels, 
          loc='upper center',
          bbox_to_anchor=(0.53, 0.69),  # 调整这个值来精确定位在第三子图下方
          ncol=3,
          fontsize=16,
          frameon=True,
          fancybox=False,
          shadow=False,
          edgecolor='black',
          framealpha=1.0)

fig.legend(method_handles, method_labels, 
          loc='upper center',
          bbox_to_anchor=(0.53, 0.49),  # 调整这个值来精确定位在第三子图下方
          ncol=3,
          fontsize=16,
          frameon=True,
          fancybox=False,
          shadow=False,
          edgecolor='black',
          framealpha=1.0)

# 属性图例（放在第四张图下方）
attribute_handles, attribute_labels = ax4.get_legend_handles_labels()
fig.legend(attribute_handles, attribute_labels,
          loc='upper center',
          bbox_to_anchor=(0.53, 0.17),  # 调整这个值来精确定位在第四子图下方
          ncol=5,
          fontsize=16,
          frameon=True,
          fancybox=False,
          shadow=False,
          edgecolor='black',
          framealpha=1.0)

# 调整子图之间的间距 - 将第2、3个子图间距减为0
plt.subplots_adjust(hspace=0.05)  # 进一步减少子图间垂直间距，使第2、3子图紧贴

# 调整整体布局，为底部图例留出空间
fig.tight_layout(rect=[0, 0.15, 1, 0.95])  # 调整底部边距为图例留空间

# 保存图表
plt.savefig('evaluation_dynamic.pdf', dpi=300, bbox_inches='tight')
plt.close()