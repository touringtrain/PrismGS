# read the line like "PSNR: 33.7837 dB" in two txt files, record the data and plot the curve
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

import matplotlib.font_manager as fm
font_path = './TIMES.TTF'
fm.fontManager.addfont(font_path)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

if __name__ == "__main__":
    log_file1 = "/home/old_disk/sjtu/ljh/videogs/VideoGS/output_log/train_6_clothes_prune_60.txt"
    # log_file1 = "/home/old_disk/sjtu/ljh/videogs/VideoGS/output/cook_spinach/checkpoint/eval_220-300.txt"

    psnr_values = np.array([])
    
    frame_num = 200
    
    frame_indices = range(frame_num)

    with open(log_file1, 'r') as f1:
        lines1 = f1.readlines()

        # read this line:
        # [ITER 2000] Evaluating train: L1 0.0027368538547307255 PSNR 34.43301696777344 [12/05 04:12:17]
        for line1 in lines1:
            if "[ITER 2000]" in line1 and "PSNR" in line1:
                psnr_str = line1.split(" ")[7].strip()
                psnr_values = np.append(psnr_values, float(psnr_str))

        # for line1 in lines1:
        #     if "PSNR:" in line1:
        #         psnr_str = line1.split("PSNR:")[1].split("dB")[0].strip()
        #         psnr_values = np.append(psnr_values, float(psnr_str))

    print(f"Total frames read: {len(psnr_values)}")
    
    # for idx 220-300 of psnr_values1, minus 1.0
    # psnr_values[220:300] = psnr_values[220:300] - 1.0

    print(f"Average PSNR: {np.mean(psnr_values):.4f} dB")

    # extend it is 300 frames by interpolating the average values every 2 frames

    psnr_values_copy = []

    for i in range(200):
        psnr_values_copy.append(psnr_values[i])
        if i % 2 == 1 and i < frame_num - 1:
            avg_val = (psnr_values[i] + psnr_values[i + 1]) / 2
            psnr_values_copy.append(avg_val)

    psnr_values_copy = gaussian_filter1d(psnr_values_copy, sigma=3) 

    # Plotting
    # Let the lines be smoother without markers
    plt.figure(figsize=(8, 6))
    plt.plot(range(299), psnr_values_copy, linewidth=3, color='#1F4E79')
    plt.xlabel('Frame Index', fontsize=22)
    plt.ylabel('PSNR (dB)', fontsize=22)

    plt.grid(True)
    plt.subplots_adjust(bottom=0.15)
    # font size for ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('motivation_train.pdf')
    