import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
baseline_methods = ['D-FCGS', 'LTS', 'V^3', 'HiCoM']
baseline_bandwidth = 100
transmission_times = np.array([1 / (baseline_bandwidth / 43.2 * 30), 
                               1 / (baseline_bandwidth / 400 * 30),
                               1 / (baseline_bandwidth / 163.7 * 30), 
                               1 / (baseline_bandwidth / 216 * 30)])
                               
                            #    1 / (baseline_bandwidth / 95 * 30)])  # Transmission Time (seconds)
# rendering_times = np.array([1 / 200, 1 / 235, 1 / 387, 1 / 274, 1 / 300])     # Rendering Time (seconds)
rendering_times = np.array([1 / 200, 1 / 235, 1 / 387, 1 / 274])
# 添加额外的操作时间（假设这些时间比较长）
# extra_operation_times = np.array([0.72, 0.025, 0.02, 0, 0.03])  # Extra Operation Time (seconds)
extra_operation_times = np.array([0.72, 0.025, 0.02, 0])  # Extra Operation Time (seconds)

# 计算总耗时
total_times = transmission_times + rendering_times + extra_operation_times

# 创建图形
fig, ax = plt.subplots(figsize=(8, 6))

# 设置柱状图的位置和宽度
x = np.arange(len(baseline_methods))
width = 0.2  # 减小柱子宽度
spacing = 0.07  # 柱子之间的间距

# 绘制柱状图
bars1 = ax.bar(x - width - spacing, transmission_times, width, label='Transmission Time', 
               color='#1f77b4', alpha=0.8, edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, rendering_times, width, label='Rendering Time', 
               color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + width + spacing, extra_operation_times, width, label='Extra Operation Time', 
               color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=0.5)

# 添加30FPS参考线（0.033秒）
ax.axhline(y=0.033, color='green', linestyle='--', linewidth=4, 
           label='30 FPS Target (0.033s)', alpha=0.7)

# 在每个柱子上方添加数值标签
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        if height > 0.001:  # 只为较大的值添加标签
            # 对于特别大的值（>0.1s），使用特殊标注
            if height > 0.15:
                ax.text(bar.get_x() + bar.get_width()/2., 0.12,
                        f'{height:.1f}s',
                        ha='center', va='bottom', fontsize=16, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
            # else:
                # ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
                #         f'{height:.3f}s',
                #         ha='center', va='bottom', fontsize=9, fontweight='bold')

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)

# 设置图形属性
ax.set_ylabel('Time per Frame (seconds)', fontsize=20)
# ax.set_title('Time Analysis for 4DGS Streaming Methods', 
            #  fontsize=22, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(baseline_methods, fontsize=20)
ax.set_ylim(0, 0.15)  # 增加空间以容纳额外操作时间

# 设置y轴刻度字体大小
ax.tick_params(axis='y', which='major', labelsize=18)

# 添加图例
ax.legend(loc='upper right', fontsize=20, framealpha=0.9)

# 添加网格以便于阅读
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# 调整布局
plt.tight_layout()

# 保存图形
plt.savefig('introduction_method_comparison.pdf', dpi=300, bbox_inches='tight')

# 打印数据摘要
print("Performance Summary for Baseline Methods:")
print("=" * 60)
for i, method in enumerate(baseline_methods):
    print(f"{method}:")
    print(f"  Transmission Time: {transmission_times[i]:.3f}s")
    print(f"  Rendering Time: {rendering_times[i]:.3f}s")
    print(f"  Extra Operation Time: {extra_operation_times[i]:.3f}s")
    print(f"  Total Time: {total_times[i]:.3f}s")
    print(f"  Equivalent FPS: {1/total_times[i]:.1f}")
    print(f"  Meets 30 FPS: {'Yes' if total_times[i] <= 0.033 else 'No'}")
    print("-" * 40)