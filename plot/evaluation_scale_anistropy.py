# read 2 Gaussian ply files, plot the distribution of scales and anisotropies
import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    ply_file1 = "/home/sjtu/ljh/videogs/VideoGS/output/cook_spinach_s_h/checkpoint/0/point_cloud/iteration_16000/point_cloud.ply"
    ply_file2 = "/home/sjtu/ljh/videogs/VideoGS/output/cook_spinach/checkpoint/0/point_cloud/iteration_16000/point_cloud.ply"

    # Load the PLY files
    pcd1 = o3d.io.read_point_cloud(ply_file1)
    pcd2 = o3d.io.read_point_cloud(ply_file2)

    # Extract Gaussian attributes, scale is in 11-13 columns (0-indexed)
    gaussians1 = np.asarray(pcd1.points)
    gaussians2 = np.asarray(pcd2.points)
    
    scales1 = gaussians1[:, 11:14]
    scales2 = gaussians2[:, 11:14]
    
    print(f"Total Gaussians in model 1: {scales1.shape[0]}")
    print(f"Total Gaussians in model 2: {scales2.shape[0]}")
    
    # calculate the mean scale of the 3 axes for each Gaussian
    scale1 = np.mean(scales1, axis=1)
    scale2 = np.mean(scales2, axis=1)

    # calculate anisotropy as max_scale / min_scale for each Gaussian ValueError: zero-size array to reduction operation maximum which has no identity
    # anisotropies1 = np.max(scales1, axis=1) / np.min(scales1, axis=1)
    # anisotropies2 = np.max(scales2, axis=1) / np.min(scales2, axis=1)

    # Plotting
    plt.figure(figsize=(12, 5))

    # Scale distribution
    plt.subplot(1, 2, 1)
    plt.hist(np.log(scale1), bins=50, alpha=0.5, label='w/ normalization')
    plt.hist(np.log(scale2), bins=50, alpha=0.5, label='w/o normalization')
    plt.xlabel('Log Scale')
    plt.ylabel('Frequency')
    plt.title('Scale Distribution')
    plt.legend()
    plt.savefig('evaluation_scale_distribution.pdf')

    # Anisotropy distribution
    plt.subplot(1, 2, 2)
    plt.hist(anisotropies1, bins=50, alpha=0.5, label='w/ normalization')
    plt.hist(anisotropies2, bins=50, alpha=0.5, label='w/o normalization')
    plt.xlabel('Anisotropy')
    plt.ylabel('Frequency')
    plt.title('Anisotropy Distribution')
    plt.legend()

    plt.tight_layout()
    plt.savefig('evaluation_scale_anisotropy.pdf')
