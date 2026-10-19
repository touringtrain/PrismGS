import os
import numpy as np
import cv2
from plyfile import PlyData
import json
import argparse

def normalize_uint8(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8), min_val, max_val

def normalize_uint8_tog(data, min_val, max_val):
    normalized = (data - min_val) / (max_val - min_val) * 255.0
    return normalized.astype(np.uint8), min_val, max_val

def normalize_uint16(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized = (data - min_val) / (max_val - min_val) * (2 ** 16 - 1)
    return normalized.astype(np.uint16), min_val, max_val

def get_ply_matrix(file_path):
    plydata = PlyData.read(file_path)
    num_vertices = len(plydata['vertex'])
    num_attributes = len(plydata['vertex'].properties)
    data_matrix = np.zeros((num_vertices, num_attributes), dtype=float)
    for i, name in enumerate(plydata['vertex'].data.dtype.names):
        data_matrix[:, i] = plydata['vertex'].data[name]
    return data_matrix

def calculate_image_size(num_points):
    image_size = 8
    while image_size * image_size < num_points:
        image_size += 8
    return image_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=95)
    parser.add_argument("--frame_end", type=int, default=115)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--ply_path", type=str, default="/data/new_disk5/wangph1/output/xyz_smalliter_group40/checkpoint")
    parser.add_argument("--output_folder", type=str, default="/data/new_disk5/wangph1/output/xyz_smalliter_group40/feature_image")
    parser.add_argument("--sh_degree", type=int, default=0)
    args = parser.parse_args()

    frame_start_init = args.frame_start
    frame_end_init = args.frame_end
    group_size = args.group_size
    interval = args.interval
    ply_path = args.ply_path
    output_folder = args.output_folder
    sh_degree = args.sh_degree
    SH_N = (sh_degree + 1) * (sh_degree + 1)
    sh_number = SH_N * 3

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    min_max_json = {}
    viewer_min_max_json = {}
    group_info_json = {}

    def searchForMaxIteration(folder):
        saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
        return max(saved_iters)


    for group in range(int((frame_end_init - frame_start_init) / group_size)):

        frame_start = group * group_size + frame_start_init
        frame_end = (group + 1) * group_size - 1 + frame_start_init

        group_info_json[str(group)] = {}
        group_info_json[str(group)]['frame_index'] = [group * group_size, (group + 1) * group_size - 1]
        group_info_json[str(group)]['name_index'] = [frame_start, frame_end]

        output_path = os.path.join(output_folder, f"group{group}")
        os.makedirs(output_path, exist_ok=True)

        for frame in range(frame_start, frame_end + 1, interval):

            png_ind = (frame - frame_start ) / interval

            ckpt_path = os.path.join(ply_path, str(frame), "point_cloud")
            # search max iteration
            max_iter = searchForMaxIteration(ckpt_path)

            # data = get_ply_matrix(os.path.join(ply_path, f"point_cloud_{frame}.ply"))
            current_data = get_ply_matrix(os.path.join(ply_path, str(frame), "point_cloud", f"iteration_{max_iter}", f"point_cloud.ply"))

            num_points = current_data.shape[0]
            image_size = calculate_image_size(num_points=num_points)
            num_attributes = current_data.shape[1]

            min_max_json[f'{frame}_num'] = num_points
            viewer_min_max_json[frame] = {}
            viewer_min_max_json[frame]['num'] = num_points
            viewer_min_max_json[frame]['info'] = []

            # rotation_data = current_data[:, -4:]
            # rotation_length = np.sqrt(np.sum(rotation_data ** 2, axis=1))
            # rotation_data_normalized = rotation_data / rotation_length[:, None]
            # current_data[:, -4:] = rotation_data_normalized
            # scale_data = np.exp(current_data[:, -7:-4])
            # current_data[:, -7:-4] = scale_data
            # opacity_data = 1 / (1 + np.exp(-current_data[:, -8]))
            # current_data[:, -8] = opacity_data
            # shs_data = current_data[:, 6:6 + sh_number].copy()
            # current_data[:, 6] = shs_data[:, 0]
            # current_data[:, 7] = shs_data[:, 1]
            # current_data[:, 8] = shs_data[:, 2]
            # # rearrange
            # for j in range(1, SH_N):
            #     current_data[:, j * 3 + 0 + 6] = shs_data[:, (j - 1) + 3]
            #     current_data[:, j * 3 + 1 + 6] = shs_data[:, (j - 1) + SH_N + 2]
            #     current_data[:, j * 3 + 2 + 6] = shs_data[:, (j - 1) + 2 * SH_N + 1]

            for i in range(num_attributes):
                if i > 2:
                    attribute_data, min_val, max_val = normalize_uint8(current_data[:, i])
                    min_max_json[f'{frame}_{i}_min'] = float(min_val)
                    min_max_json[f'{frame}_{i}_max'] = float(max_val)
                    viewer_min_max_json[frame]['info'].append(float(min_val))
                    viewer_min_max_json[frame]['info'].append(float(max_val))
                    attribute_data_reshaped = attribute_data.reshape(-1, 1)
                    image = np.zeros((image_size * image_size, 1), dtype=np.uint8)
                    image[:attribute_data_reshaped.shape[0], :] = attribute_data_reshaped  
                    image_reshaped = image.reshape((image_size, image_size))
                    cv2.imwrite(os.path.join(output_path, f"{frame}_{i+3}.png"), image_reshaped)
                else: 
                    attribute_data, min_val, max_val = normalize_uint16(current_data[:, i])
                    min_max_json[f'{frame}_{i}_min'] = float(min_val)
                    min_max_json[f'{frame}_{i}_max'] = float(max_val)
                    viewer_min_max_json[frame]['info'].append(float(min_val))
                    viewer_min_max_json[frame]['info'].append(float(max_val))
                    attribute_data_reshaped = attribute_data.reshape(-1, 1)
                    image_odd = np.zeros((image_size * image_size, 1), dtype=np.uint8)
                    image_even = np.zeros((image_size * image_size, 1), dtype=np.uint8)
                    #split the uint16 into two uint8, one is all the odd bits, the other is all the even bits
                    # for j in range(16):
                    #     if j % 2 == 0:
                    #         image_even[:attribute_data_reshaped.shape[0], :] += ((attribute_data_reshaped >> j) & 1) << (j // 2)
                    #     else:
                    #         image_odd[:attribute_data_reshaped.shape[0], :] += ((attribute_data_reshaped >> j) & 1) << (j // 2)
                    
                    image_even[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped & 0xff)
                    image_odd[:attribute_data_reshaped.shape[0], :] += (attribute_data_reshaped >> 8)

                    image_odd_reshaped = image_odd.reshape((image_size, image_size))
                    image_even_reshaped = image_even.reshape((image_size, image_size))
                    cv2.imwrite(os.path.join(output_path, f"{frame}_{2*i}.png"), image_even_reshaped)
                    cv2.imwrite(os.path.join(output_path, f"{frame}_{2*i+1}.png"), image_odd_reshaped)

    with open(os.path.join(output_folder, "min_max.json"), "w") as f:
        json.dump(min_max_json, f, indent=4)

    with open(os.path.join(output_folder, "viewer_min_max.json"), "w") as f:
        json.dump(viewer_min_max_json, f, indent=4)

    with open(os.path.join(output_folder, "group_info.json"), "w") as f:
        json.dump(group_info_json, f, indent=4)