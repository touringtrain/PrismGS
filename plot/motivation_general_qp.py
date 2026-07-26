import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

# 移除中文字体设置（Linux服务器无需），直接使用默认英文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# 准备数据（保持不变）
qp_values = [0, 5, 10, 15, 20, 25, 30, 35, 40]
size_mb = [184, 183, 141, 106, 80, 67, 58, 50, 42]
perframe_bitrate_kb = np.array(size_mb) * 1000 / 200  # 原始计算保持不变
bitrate_kbps = perframe_bitrate_kb  # 转换为30FPS的比特率 (KB/s)
psnr = [36.2693, 36.2334, 36.0995, 35.7345, 34.7456, 32.3418, 27.9024, 22.3123, 17.4712]
ssim = [0.9909, 0.9907, 0.9899, 0.9879, 0.9831, 0.9716, 0.9467, 0.9079, 0.8587]

# 创建第一张图：PSNR和SSIM
fig1, ax1 = plt.subplots(figsize=(9, 6))

# 绘制PSNR（左侧y轴）
color_psnr = 'tab:red'
ax1.set_xlabel('QP Value', fontsize=28)
ax1.set_ylabel('PSNR (dB)', color=color_psnr, fontsize=28)
ax1.plot(qp_values, psnr, 's-', color=color_psnr, linewidth=3, markersize=8, label='PSNR')
ax1.tick_params(axis='y', labelcolor=color_psnr)
ax1.set_ylim(15, 40)  # 固定PSNR范围
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax1.grid(True, alpha=0.3)

# 创建右侧y轴：绘制SSIM
ax2 = ax1.twinx()  
color_ssim = 'tab:green'
ax2.set_ylabel('SSIM', color=color_ssim, fontsize=28)
ax2.plot(qp_values, ssim, '^--', color=color_ssim, linewidth=3, markersize=8, label='SSIM')
ax2.tick_params(axis='y', labelcolor=color_ssim)
ax2.set_ylim(0.75, 1.0)  # 固定SSIM范围
ax2.grid(True, alpha=0.3)

# 调整刻度字号
ax1.tick_params(axis='x', labelsize=28)
ax1.tick_params(axis='y', labelsize=28)
ax2.tick_params(axis='y', labelsize=28)

# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', framealpha=0.9, fontsize=24)

# 调整布局
fig1.tight_layout()
plt.savefig('motivation_general_quality_metrics.pdf', bbox_inches='tight')
plt.close()

# 创建第二张图：比特率
fig2, ax3 = plt.subplots(figsize=(8, 6))

# 绘制比特率
color_bitrate = 'tab:blue'
ax3.set_xlabel('QP Value', fontsize=28)
ax3.set_ylabel('Frame Size (MB)', color=color_bitrate, fontsize=28)
ax3.plot(qp_values, bitrate_kbps / 1000, 'o-', color=color_bitrate, linewidth=3, markersize=8, label='Frame Size (MB)')
ax3.tick_params(axis='y', labelcolor=color_bitrate)
ax3.xaxis.set_major_locator(MaxNLocator(integer=True))
ax3.grid(True, alpha=0.3)

# 调整刻度字号
ax3.tick_params(axis='x', labelsize=28)
ax3.tick_params(axis='y', labelsize=28)

# 添加图例
ax3.legend(loc='upper right', framealpha=0.9, fontsize=28)

# 调整布局
fig2.tight_layout()
plt.savefig('motivation_general_bitrate_analysis.pdf', bbox_inches='tight')
plt.close()
