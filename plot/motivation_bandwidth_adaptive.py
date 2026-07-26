import csv

# read the first 200 element of "Throughput" column from the CSV file
def read_throughput_data(file_path):
    throughput_data = []
    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            if i < 268:
                continue
            if i >= 318:
                break
            throughput_data.append(float(row['Throughput']))
    return throughput_data

throughput_data = read_throughput_data('../motivation_study/Lumos5G-v1.0.csv')

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

fps_static = [] 

for i in range(len(throughput_data)):
    fps_static.append(min(throughput_data[i] / 8 / 0.78 * 1.5, 30))

# print(fps_static)

# using a naive bandwidth-adaptive method

quality_coefficient = [36.27 / 35.73, 36.23 / 35.73, 36.09 / 35.73, 1, 34.74 / 35.73, 32.34 / 35.73]

size_coefficient = [240 / 156, 240 / 156, 196 / 156, 1, 128 / 156, 108 / 156]

def adaptive(throughput_data, psnr_data, quality_coefficient, size_coefficient):
    fps_adaptive = []
    psnr_adaptive = []
    for i in range(len(throughput_data)):
        max_fps = (throughput_data[i] / 8 / 0.78 * 1.5) / size_coefficient[-1]
        fps_adaptive.append(min(max_fps, 30))
        if max_fps < 30:
            psnr_adaptive.append(psnr_data[i] * quality_coefficient[-1])
        else:
            for j in range(len(size_coefficient)):
                if max_fps >= 30 / size_coefficient[j]:
                    # print(max_fps)
                    psnr_adaptive.append(psnr_data[i] * quality_coefficient[j])
                    break
    return fps_adaptive, psnr_adaptive

fps_adaptive, psnr_adaptive = adaptive(throughput_data, psnr_data, quality_coefficient, size_coefficient)
# print(fps_adaptive)
# print(psnr_adaptive)

print("Average FPS (Static Method): ", sum(fps_static) / len(fps_static), "Variance: ", sum((x - sum(fps_static) / len(fps_static)) ** 2 for x in fps_static) / len(fps_static))
print("Average FPS (Adaptive Method): ", sum(fps_adaptive) / len(fps_adaptive), "Variance: ", sum((x - sum(fps_adaptive) / len(fps_adaptive)) ** 2 for x in fps_adaptive) / len(fps_adaptive))

print("Average PSNR (Static Method): ", sum(psnr_data) / len(psnr_data), "Variance: ", sum((x - sum(psnr_data) / len(psnr_data)) ** 2 for x in psnr_data) / len(psnr_data))
print("Average PSNR (Adaptive Method): ", sum(psnr_adaptive) / len(psnr_adaptive), "Variance: ", sum((x - sum(psnr_adaptive) / len(psnr_adaptive)) ** 2 for x in psnr_adaptive) / len(psnr_adaptive))

# plot the fps and psnr data separately
import matplotlib.pyplot as plt

# plot FPS data
plt.figure(figsize=(8, 6))
plt.plot(fps_static, label='Static Method', color='blue', linestyle='--')
plt.plot(fps_adaptive, label='Adaptive Method', color='orange', linestyle='--')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('FPS', fontsize=18)
plt.title('FPS Comparison between Static and Adaptive Methods', fontsize=16)
plt.ylim(18, 32)
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('motivation_bandwidth_adaptive_fps.pdf', dpi=300)
plt.close()

# plot PSNR data
plt.figure(figsize=(8, 6))
plt.plot(psnr_data, label='Static Method', color='blue', linestyle='--')
plt.plot(psnr_adaptive, label='Adaptive Method', color='orange', linestyle='--')
plt.xlabel('Time (s)', fontsize=18)
plt.ylabel('PSNR (dB)', fontsize=18)
plt.title('PSNR Comparison between Static and Adaptive Methods', fontsize=16)
plt.ylim(30, 36)    
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(fontsize=18)
plt.tight_layout()
plt.savefig('motivation_bandwidth_adaptive_psnr.pdf', dpi=300)
plt.close()

# plot FPS and PSNR for static method, left y-axis is FPS, right y-axis is PSNR, with different ylim
plt.figure(figsize=(8, 6))

# 创建第一个y轴（左侧）
ax1 = plt.gca()
ax1.plot(fps_static, label='FPS', color='blue', linestyle='--', linewidth=3)
ax1.set_xlabel('Frame', fontsize=22)
ax1.set_ylabel('FPS', fontsize=22)
ax1.set_ylim(18, 36)  # 设置左侧y轴范围

# 创建第二个y轴（右侧）
ax2 = ax1.twinx()
ax2.plot(psnr_data, label='PSNR', color='orange', linestyle='--', linewidth=3)
ax2.set_ylabel('PSNR (dB)', fontsize=22)
ax2.set_ylim(30, 40)  # 设置右侧y轴范围

plt.title('FPS and PSNR for Static Method', fontsize=20)
plt.grid(True, alpha=0.3, linestyle='--')

# set the fontsize of the axis tick labels
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

# embed a table for mean and variance of fps_static and psnr_data
mean_fps_static = sum(fps_static) / len(fps_static)
var_fps_static = sum((x - mean_fps_static) ** 2 for x in fps_static) / len(fps_static)
mean_psnr_data = sum(psnr_data) / len(psnr_data)
var_psnr_data = sum((x - mean_psnr_data) ** 2 for x in psnr_data) / len(psnr_data)
table_data = [
    ['Metric', 'Mean', 'Variance'],
    ['FPS', f'{mean_fps_static:.2f}', f'{var_fps_static:.2f}'],
    ['PSNR', f'{mean_psnr_data:.2f}', f'{var_psnr_data:.2f}']
]
table = plt.table(cellText=table_data, colLabels=None, cellLoc='center',
                  loc='upper left', colWidths=[0.16, 0.16, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(20)
table.scale(1.2, 2)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=22)

plt.tight_layout()
plt.savefig('motivation_bandwidth_adaptive_static.pdf', dpi=300)
plt.close()


# plot FPS and PSNR for adaptive method, left y-axis is FPS, right y-axis is PSNR
plt.figure(figsize=(8, 6))
# 创建第一个y轴（左侧）
ax1 = plt.gca()
ax1.plot(fps_adaptive, label='FPS', color='blue', linestyle='--', linewidth=3)
ax1.set_xlabel('Frame', fontsize=22)
ax1.set_ylabel('FPS', fontsize=22)
ax1.set_ylim(18, 36)  # 设置左侧y轴范围 
# 创建第二个y轴（右侧）
ax2 = ax1.twinx()
ax2.plot(psnr_adaptive, label='PSNR', color='orange', linestyle='--', linewidth=3)
ax2.set_ylabel('PSNR (dB)', fontsize=22)
ax2.set_ylim(30, 40)  # 设置右侧y轴范围
plt.title('FPS and PSNR for Adaptive Method', fontsize=20)
plt.grid(True, alpha=0.3, linestyle='--')

# set the fontsize of the axis tick labels
ax1.tick_params(axis='both', which='major', labelsize=20)
ax2.tick_params(axis='both', which='major', labelsize=20)

# embed a table for mean and variance of fps_static and psnr_data
mean_fps_adaptive = sum(fps_adaptive) / len(fps_adaptive)
var_fps_adaptive = sum((x - mean_fps_adaptive) ** 2 for x in fps_adaptive) / len(fps_adaptive)
mean_psnr_adaptive = sum(psnr_adaptive) / len(psnr_adaptive)
var_psnr_adaptive = sum((x - mean_psnr_adaptive) ** 2 for x in psnr_adaptive) / len(psnr_adaptive)
table_data = [
    ['Metric', 'Mean', 'Variance'],
    ['FPS', f'{mean_fps_adaptive:.2f}', f'{var_fps_adaptive:.2f}'],
    ['PSNR', f'{mean_psnr_adaptive:.2f}', f'{var_psnr_adaptive:.2f}']
]
table = plt.table(cellText=table_data, colLabels=None, cellLoc='center',
                  loc='upper left', colWidths=[0.16, 0.16, 0.18])
table.auto_set_font_size(False)
table.set_fontsize(20)
table.scale(1.2, 2)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=22)
plt.tight_layout()
plt.savefig('motivation_bandwidth_adaptive_adaptive.pdf', dpi=300)
plt.close()
