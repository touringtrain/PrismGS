# import os
# import numpy as np
# import plyfile
# import json
# import cv2
# import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument("--input_path", type=str, default="../output/4k_4_dancing_prune")
# parser.add_argument("--output_path", type=str, default="4k_4_dancing")
# parser.add_argument("--start", type=int, default=0)
# parser.add_argument("--end", type=int, default=200)
# parser.add_argument("--group_size", type=int, default=200)
# parser.add_argument("--qp", type=str, default="25")

# args = parser.parse_args()

# # input_path = "/data/new_disk5/wangph1/output/xyz_smalliter_group40"
# input_path = args.input_path

# start = args.start
# end =  args.end
# group_size = args.group_size
# interval = 1
# qp = args.qp
# sh_degree = 0
# SH_N = (sh_degree + 1) * (sh_degree + 1)
# sh_number = SH_N * 3
# num_video = 20
# # output_path = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/decompress/qp_{qp}"
# output_path = f"{args.output_path}/qp_{qp}"
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# feature_video_path = os.path.join(input_path, f"feature_video", f"png_all_{qp}")
# feature_image_path = os.path.join(input_path, "feature_image")
# min_max_path = os.path.join(feature_image_path, "min_max.json")

# group_idx = 0

# def read_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     frames = []
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frames.append(gray_frame)
#     return frames

# def get_attribute():
#     attribute_names = []
#     attribute_names.append('x')
#     attribute_names.append('y')
#     attribute_names.append('z')
#     attribute_names.append('nx')
#     attribute_names.append('ny')
#     attribute_names.append('nz')
#     for i in range(3):
#         attribute_names.append('f_dc_' + str(i))
#     for i in range(45):
#         attribute_names.append('f_rest_' + str(i))
#     attribute_names.append('opacity')
#     for i in range(3):
#         attribute_names.append('scale_' + str(i))
#     for i in range(4):
#         attribute_names.append('rot_' + str(i))

#     return attribute_names

# def denormalize_uint8(data, min_val, max_val):
#     return data / 255.0 * (max_val - min_val) + min_val

# def denormalize_uint16(data, min_val, max_val):
#     return data / (2 ** 16 - 1) * (max_val - min_val) + min_val

# def reconstruct_ply_from_images(frame, num_attributes, image_size, input_folder, min_max_info):
#     reconstructed_data = np.zeros((image_size * image_size, num_attributes), dtype=float)
    
#     for i in range(num_attributes):
#         img_path = os.path.join(input_folder, f"{frame}_{i}.png")
#         img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
#         min_val = float(min_max_info[f'{frame}_{i}_min'])
#         max_val = float(min_max_info[f'{frame}_{i}_max'])
        
#         img_denormalized = denormalize_uint16(img, min_val, max_val)
        
#         reconstructed_data[:, i] = img_denormalized.flatten()
        
#     actual_num_points = min_max_info[f'{frame}_num']
#     reconstructed_data = reconstructed_data[:actual_num_points]
    
#     return reconstructed_data, actual_num_points

# def save_ply(residual, output_file):
#     n, k = residual.shape

#     attribute_names = []
#     attribute_names.append('x')
#     attribute_names.append('y')
#     attribute_names.append('z')
#     attribute_names.append('nx')
#     attribute_names.append('ny')
#     attribute_names.append('nz')
#     for i in range(3):
#         attribute_names.append('f_dc_' + str(i))
#     # for i in range(sh_number):
#     #     attribute_names.append('f_rest_' + str(i))
#     attribute_names.append('opacity')
#     for i in range(3):
#         attribute_names.append('scale_' + str(i))
#     for i in range(4):
#         attribute_names.append('rot_' + str(i))

#     assert k == len(attribute_names)

#     with open(output_file, 'wb') as ply_file:
#         ply_file.write(b"ply\n")
#         ply_file.write(b"format binary_little_endian 1.0\n")
#         ply_file.write(b"element vertex %d\n" % n)
        
#         for attribute_name in attribute_names:
#             ply_file.write(b"property float %s\n" % attribute_name.encode())
        
#         ply_file.write(b"end_header\n")
        
#         for i in range(n):
#             vertex_data = residual[i].astype(np.float32).tobytes()
#             ply_file.write(vertex_data)

# import time

# start_time = time.time()

# with open(min_max_path, "r") as f:
#     min_max_info = json.load(f)

# for frame in range(start, end, group_size * interval):
#     group_start = frame
#     group_end = min(frame + group_size * interval, end)
#     print(group_start, group_end)

#     group_video_path = os.path.join(feature_video_path, f"group{group_idx}")
#     group_video_data = []
#     for video_idx in range(num_video):
#         video_path = os.path.join(group_video_path, f"{video_idx}.mp4")
#         frames = read_video(video_path)
#         group_video_data.append(frames)
#     group_idx += 1
#     group_frame_idx = 0
#     # reconstruct a group
#     for group_frame in range(group_start, group_end, interval):
#         group_frame_data = np.zeros((min_max_info[f'{group_frame}_num'], num_video - 3), dtype=float)
#         # position
#         for att in range(3):            # concat uint8 to uint16
#             image_even = group_video_data[att * 2][group_frame_idx]
#             image_odd = group_video_data[att * 2 + 1][group_frame_idx]
#             # image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
#             # image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
#             image_even = image_even.astype(np.uint16)
#             image_odd = image_odd.astype(np.uint16)
#             image = image_even + (image_odd << 8)
#             # denormalize
#             min_val = float(min_max_info[f'{group_frame}_{att}_min'])
#             max_val = float(min_max_info[f'{group_frame}_{att}_max'])
#             # print(denormalize_uint16(image, min_val, max_val).shape)
#             # print(group_frame_data[:, att].shape)
#             group_frame_data[:, att] = denormalize_uint16(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]
#         for att in range(3, 17):
#             if att in [3, 4, 5]:
#                 continue
#             image = group_video_data[att + 3][group_frame_idx]
#             # denormalize
#             min_val = float(min_max_info[f'{group_frame}_{att}_min'])
#             max_val = float(min_max_info[f'{group_frame}_{att}_max'])
#             group_frame_data[:, att] = denormalize_uint8(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]

#         # save ply
#         # if the directory does not exist, create it
#         # group_frame_path = os.path.join(output_path, f"{group_frame}", "point_cloud", "iteration_1")
#         # if not os.path.exists(group_frame_path):
#         #     os.makedirs(group_frame_path)
#         # save_ply(group_frame_data, os.path.join(group_frame_path, "point_cloud.ply"))

#         group_frame_idx += 1

# end_time = time.time()
# print("Total decompression time:", end_time - start_time, "seconds")

import os
import numpy as np
import json
import cv2
import argparse
import torch
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# --- 参数设定 ---
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, default="../output/4k_4_dancing_prune")
parser.add_argument("--output_path", type=str, default="4k_4_dancing")
parser.add_argument("--start", type=int, default=0)
parser.add_argument("--end", type=int, default=200)
parser.add_argument("--group_size", type=int, default=200)
parser.add_argument("--qp", type=str, default="25")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_path = args.input_path
start, end, group_size = args.start, args.end, args.group_size
qp = args.qp
num_video = 20
output_path = f"{args.output_path}/qp_{qp}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

feature_video_path = os.path.join(input_path, f"feature_video", f"png_all_{qp}")
feature_image_path = os.path.join(input_path, "feature_image")
min_max_path = os.path.join(feature_image_path, "min_max.json")

# --- 辅助函数 ---

def read_video_to_cpu(video_path):
    """ 在后台线程中将视频读入 CPU 内存 """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    return np.array(frames) # [Frames, H, W]

def load_group_videos(group_idx):
    """ 多线程预加载一个组的所有 20 个视频 """
    group_video_path = os.path.join(feature_video_path, f"group{group_idx}")
    video_paths = [os.path.join(group_video_path, f"{i}.mp4") for i in range(num_video)]
    
    # 使用线程池并发读取磁盘上的 20 个 MP4 文件
    with ThreadPoolExecutor(max_workers=8) as executor:
        group_data = list(executor.map(read_video_to_cpu, video_paths))
    
    # 将 numpy 数组转为 GPU Tensor
    gpu_group_data = [torch.from_numpy(v).to(device, non_blocking=True) for v in group_data]
    return gpu_group_data

def denormalize_gpu(data, min_val, max_val, bits=8):
    denom = (2**bits - 1)
    return data.float() / denom * (max_val - min_val) + min_val

def save_ply_gpu(residual_gpu, output_file):
    """ 异步保存：不阻塞主解码循环 """
    residual = residual_gpu.cpu().numpy()
    n = residual.shape[0]
    attribute_names = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'opacity', 
                       'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3']
    
    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\nformat binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        for name in attribute_names:
            ply_file.write(b"property float %s\n" % name.encode())
        ply_file.write(b"end_header\n")
        ply_file.write(residual.astype(np.float32).tobytes())

# --- 主解码流程 ---

with open(min_max_path, "r") as f:
    min_max_info = json.load(f)

# 队列用于预取 (Prefetching)
group_queue = Queue(maxsize=2)

def prefetch_worker():
    """ 预取线程：提前加载下一组视频 """
    current_idx = 0
    for frame_s in range(start, end, group_size):
        group_queue.put(load_group_videos(current_idx))
        current_idx += 1

import threading
prefetch_thread = threading.Thread(target=prefetch_worker)
prefetch_thread.start()

start_time = time.time()
group_count = 0

for frame_start in range(start, end, group_size):
    group_end = min(frame_start + group_size, end)
    
    # 从队列获取已经加载好的 GPU 组数据
    group_video_data = group_queue.get()
    
    print(f"Processing Group {group_count} on GPU with TRUE parallel decoding...")
    
    num_frames_in_group = group_end - frame_start
    
    # 预先获取所有帧的点数，找到最大点数用于创建统一batch
    num_pts_list = []
    for f_idx in range(num_frames_in_group):
        current_frame = frame_start + f_idx
        num_pts_list.append(min_max_info[f'{current_frame}_num'])
    
    max_pts = max(num_pts_list)
    
    # ========== 真正的并行：创建 [num_frames, max_pts, 17] 的3D tensor ==========
    # 初始化所有帧的数据（一次性分配大tensor）
    all_frames_data = torch.zeros((num_frames_in_group, max_pts, 17), device=device)
    
    # 创建mask标记有效点（避免padding干扰）
    valid_mask = torch.zeros((num_frames_in_group, max_pts), dtype=torch.bool, device=device)
    for f_idx, num_pts in enumerate(num_pts_list):
        valid_mask[f_idx, :num_pts] = True
    
    # 1. 并行处理位置 XYZ (uint16) - 所有帧、所有点同时处理
    for att in range(3):
        # 创建 [num_frames, max_pts] 的batch tensor
        batch_combined = torch.zeros((num_frames_in_group, max_pts), dtype=torch.int32, device=device)
        
        # 收集所有帧的数据（这里仍然是逐帧，但数据复制很快）
        for f_idx in range(num_frames_in_group):
            current_frame = frame_start + f_idx
            img_even = group_video_data[att * 2][f_idx].to(torch.int32)
            img_odd = group_video_data[att * 2 + 1][f_idx].to(torch.int32)
            combined = img_even + (img_odd << 8)
            
            num_pts = num_pts_list[f_idx]
            batch_combined[f_idx, :num_pts] = combined.view(-1)[:num_pts]
        
        # 获取所有帧的min/max值
        min_vals = torch.zeros(num_frames_in_group, device=device)
        max_vals = torch.zeros(num_frames_in_group, device=device)
        for f_idx in range(num_frames_in_group):
            current_frame = frame_start + f_idx
            min_vals[f_idx] = float(min_max_info[f'{current_frame}_{att}_min'])
            max_vals[f_idx] = float(min_max_info[f'{current_frame}_{att}_max'])
        
        # **关键：批量反归一化，所有帧同时处理**
        denom = (2**16 - 1)
        # 扩展维度：[num_frames, 1] 广播到 [num_frames, max_pts]
        min_tensor = min_vals.unsqueeze(1)
        max_tensor = max_vals.unsqueeze(1)
        
        batch_denorm = batch_combined.float() / denom * (max_tensor - min_tensor) + min_tensor
        
        all_frames_data[:, :, att] = batch_denorm
    
    # 2. 并行处理其他属性 (uint8) - 同样使用batch操作
    for att in range(3, 17):
        if att in [3, 4, 5]:
            continue
        
        # 创建 [num_frames, max_pts] 的batch tensor
        batch_img = torch.zeros((num_frames_in_group, max_pts), dtype=torch.uint8, device=device)
        
        for f_idx in range(num_frames_in_group):
            current_frame = frame_start + f_idx
            img = group_video_data[att + 3][f_idx]
            
            num_pts = num_pts_list[f_idx]
            batch_img[f_idx, :num_pts] = img.view(-1)[:num_pts]
        
        # 获取min/max值
        min_vals = torch.zeros(num_frames_in_group, device=device)
        max_vals = torch.zeros(num_frames_in_group, device=device)
        for f_idx in range(num_frames_in_group):
            current_frame = frame_start + f_idx
            min_vals[f_idx] = float(min_max_info[f'{current_frame}_{att}_min'])
            max_vals[f_idx] = float(min_max_info[f'{current_frame}_{att}_max'])
        
        # **批量反归一化：所有帧同时处理**
        denom = (2**8 - 1)
        min_tensor = min_vals.unsqueeze(1)
        max_tensor = max_vals.unsqueeze(1)
        
        batch_denorm = batch_img.float() / denom * (max_tensor - min_tensor) + min_tensor
        
        # 写入结果
        all_frames_data[:, :, att] = batch_denorm
    
    # 3. 保存解码后的帧（如果需要）
    for f_idx in range(num_frames_in_group):
        current_frame = frame_start + f_idx
        num_pts = num_pts_list[f_idx]
        
        # 提取有效数据
        frame_data = all_frames_data[f_idx, :num_pts, :]
        
        out_frame_path = os.path.join(output_path, f"{current_frame}", "point_cloud", "iteration_1")
        if not os.path.exists(out_frame_path):
            os.makedirs(out_frame_path)
        
        # 取消注释以保存
        # save_ply_gpu(frame_data, os.path.join(out_frame_path, "point_cloud.ply"))

    group_count += 1

prefetch_thread.join()
end_time = time.time()
print(f"Total Multithreaded GPU Decompression: {end_time - start_time:.4f} seconds")


