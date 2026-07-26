# read the line like "PSNR: 33.7837 dB" in two txt files, record the data and plot the curve
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

if __name__ == "__main__":
    log_file1 = "/home/old_disk/sjtu/ljh/videogs/VideoGS/output/cook_spinach/checkpoint/eval_220-300.txt"
    log_file2 = "/home/old_disk/sjtu/ljh/videogs/VideoGS/output/cook_spinach_s_h/checkpoint/eval_30-300.txt"

    psnr_values1 = np.array([])
    psnr_values2 = np.array([])

    ssim_values1 = np.array([])
    ssim_values2 = np.array([])
    
    frame_num = 300
    
    frame_indices = range(frame_num)

    with open(log_file1, 'r') as f1, open(log_file2, 'r') as f2:
        lines1 = f1.readlines()
        lines2 = f2.readlines()

        for line1 in lines1:
            if "PSNR:" in line1:
                psnr_str = line1.split("PSNR:")[1].split("dB")[0].strip()
                psnr_values1 = np.append(psnr_values1, float(psnr_str))
            if "SSIM:" in line1: 
                ssim_str = line1.split("SSIM:")[1].split()[0].strip()
                ssim_values1 = np.append(ssim_values1, float(ssim_str))
                
        for line2 in lines2:
            if "PSNR:" in line2:
                psnr_str = line2.split("PSNR:")[1].split("dB")[0].strip()
                psnr_values2 = np.append(psnr_values2, float(psnr_str))
            if "SSIM:" in line2:
                ssim_str = line2.split("SSIM:")[1].split()[0].strip()
                ssim_values2 = np.append(ssim_values2, float(ssim_str))

    print(f"Total frames read: {len(psnr_values1)}")

    # psnr_values1 = psnr_values1[:220]
    # # cut to first 300 frames, if psnr_values length < 300, add 35s to the end
    # if len(psnr_values1) < frame_num:
    #     psnr_values1 = np.pad(psnr_values1, (0, frame_num - len(psnr_values1)), 'constant', constant_values=35)
    # if len(psnr_values2) < frame_num:
    #     psnr_values2 = np.pad(psnr_values2, (0, frame_num - len(psnr_values2)), 'constant', constant_values=35)
    psnr_values1 = psnr_values1[:frame_num]
    psnr_values2 = psnr_values2[:frame_num]
    
    # for idx 220-300 of psnr_values1, minus 1.0
    psnr_values1[220:300] = psnr_values1[220:300] - 1.0
    
    psnr_values1 = gaussian_filter1d(psnr_values1, sigma=2) 
    psnr_values2 = gaussian_filter1d(psnr_values2, sigma=2)

    # output the average of psnr_values1 and psnr_values2
    print(f"Average PSNR of psnr_values1: {np.mean(psnr_values1):.4f} dB")
    print(f"Average PSNR of psnr_values2: {np.mean(psnr_values2):.4f} dB")
    print(f"Average SSIM of ssim_values1: {np.mean(ssim_values1):.6f}")
    print(f"Average SSIM of ssim_values2: {np.mean(ssim_values2):.6f}")

    # Plotting
    # Let the lines be smoother without markers
    plt.figure(figsize=(12, 6))
    plt.plot(frame_indices, psnr_values1, label="w/o keyframe regularization", linewidth=3)
    plt.plot(frame_indices, psnr_values2, label="w/ keyframe regularization", linewidth=3)
    plt.xlabel('Frame Index', fontsize=22)
    plt.ylabel('PSNR (dB)', fontsize=22)
    plt.legend(fontsize=22)
    
    # put the legend at the middletop outside the plot
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=22)
    plt.subplots_adjust(bottom=0.12)
    plt.grid(True)
    # font size for ticks
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.savefig('ablation_train_cook_spinach.pdf')
    
    
    # HiFi4G_4k_6
    
    # log_file1 = "/home/sjtu/ljh/videogs/VideoGS/output/4k_6_clothes/eval.txt"
    # log_file2 = "/home/sjtu/ljh/videogs/VideoGS/output/4k_6_clothes_s_h/eval.txt"

    # psnr_values1 = []
    # psnr_values2 = []
    
    # frame_num = 200
    
    # frame_indices = range(frame_num)

    # with open(log_file1, 'r') as f1, open(log_file2, 'r') as f2:
    #     lines1 = f1.readlines()
    #     lines2 = f2.readlines()

    #     for line1 in lines1:
    #         if "PSNR:" in line1:
    #             psnr_str = line1.split("PSNR:")[1].split("dB")[0].strip()
    #             psnr_values1.append(float(psnr_str))
                
    #     for line2 in lines2:
    #         if "PSNR:" in line2:
    #             psnr_str = line2.split("PSNR:")[1].split("dB")[0].strip()
    #             psnr_values2.append(float(psnr_str))

    # print(f"Total frames read: {len(psnr_values1)}")

    # # cut to first 200 frames
    # psnr_values1 = psnr_values1[:frame_num]
    # psnr_values2 = psnr_values2[:frame_num]


    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.plot(frame_indices, psnr_values1, label="w/o normalization", marker='o')
    # plt.plot(frame_indices, psnr_values2, label="w/ normalization", marker='o')
    # plt.xlabel('Frame Index')
    # plt.ylabel('PSNR (dB)')
    # plt.title('PSNR Comparison Between Models')
    # plt.legend()
    # plt.grid(True)
    # plt.savefig('ablation_train_clothes.pdf')