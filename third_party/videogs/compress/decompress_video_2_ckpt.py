import os
import numpy as np
import plyfile
import json
import cv2

input_path = "/data/new_disk5/wangph1/output/xyz_smalliter_group40"

start = 95
end = 115
group_size = 20
interval = 1
qp = "manual"
sh_degree = 0
SH_N = (sh_degree + 1) * (sh_degree + 1)
sh_number = SH_N * 3
num_video = 20
output_path = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/decompress/qp_{qp}"
if not os.path.exists(output_path):
    os.makedirs(output_path)

feature_video_path = os.path.join(input_path, "feature_video", f"png_all_{qp}")
feature_image_path = os.path.join(input_path, "feature_image")
min_max_path = os.path.join(feature_image_path, "min_max.json")

group_idx = 0

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray_frame)
    return frames

def get_attribute():
    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    for i in range(45):
        attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    return attribute_names

def denormalize_uint8(data, min_val, max_val):
    return data / 255.0 * (max_val - min_val) + min_val

def denormalize_uint16(data, min_val, max_val):
    return data / (2 ** 16 - 1) * (max_val - min_val) + min_val

def reconstruct_ply_from_images(frame, num_attributes, image_size, input_folder, min_max_info):
    reconstructed_data = np.zeros((image_size * image_size, num_attributes), dtype=float)
    
    for i in range(num_attributes):
        img_path = os.path.join(input_folder, f"{frame}_{i}.png")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        
        min_val = float(min_max_info[f'{frame}_{i}_min'])
        max_val = float(min_max_info[f'{frame}_{i}_max'])
        
        img_denormalized = denormalize_uint16(img, min_val, max_val)
        
        reconstructed_data[:, i] = img_denormalized.flatten()
        
    actual_num_points = min_max_info[f'{frame}_num']
    reconstructed_data = reconstructed_data[:actual_num_points]
    
    return reconstructed_data, actual_num_points

def save_ply(residual, output_file):
    n, k = residual.shape

    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    # for i in range(sh_number):
    #     attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    assert k == len(attribute_names)

    with open(output_file, 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % n)
        
        for attribute_name in attribute_names:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        
        for i in range(n):
            vertex_data = residual[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)

with open(min_max_path, "r") as f:
    min_max_info = json.load(f)

for frame in range(start, end, group_size * interval):
    group_start = frame
    group_end = min(frame + group_size * interval, end)
    print(group_start, group_end)

    group_video_path = os.path.join(feature_video_path, f"group{group_idx}")
    group_video_data = []
    for video_idx in range(num_video):
        video_path = os.path.join(group_video_path, f"{video_idx}.mp4")
        frames = read_video(video_path)
        group_video_data.append(frames)
    group_idx += 1

    group_frame_idx = 0
    # reconstruct a group
    for group_frame in range(group_start, group_end, interval):
        group_frame_data = np.zeros((min_max_info[f'{group_frame}_num'], num_video - 3), dtype=float)
        # position
        for att in range(3):
            # concat uint8 to uint16
            image_even = group_video_data[att * 2][group_frame_idx]
            image_odd = group_video_data[att * 2 + 1][group_frame_idx]
            # image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
            # image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)
            image_even = image_even.astype(np.uint16)
            image_odd = image_odd.astype(np.uint16)
            image = image_even + (image_odd << 8)
            # denormalize
            min_val = float(min_max_info[f'{group_frame}_{att}_min'])
            max_val = float(min_max_info[f'{group_frame}_{att}_max'])
            # print(denormalize_uint16(image, min_val, max_val).shape)
            # print(group_frame_data[:, att].shape)
            group_frame_data[:, att] = denormalize_uint16(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]
        for att in range(3, 17):
            if att in [3, 4, 5]:
                continue
            image = group_video_data[att + 3][group_frame_idx]
            # denormalize
            min_val = float(min_max_info[f'{group_frame}_{att}_min'])
            max_val = float(min_max_info[f'{group_frame}_{att}_max'])
            group_frame_data[:, att] = denormalize_uint8(image, min_val, max_val).flatten()[:min_max_info[f'{group_frame}_num']]

        # save ply
        save_ply(group_frame_data, os.path.join(output_path, f"{group_frame}.ply"))

        group_frame_idx += 1
