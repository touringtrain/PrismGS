import os
import time
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_start", type=int, default=95)
    parser.add_argument("--frame_end", type=int, default=115)
    parser.add_argument("--group_size", type=int, default=20)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--qp", type=int, default=25)

    args = parser.parse_args()
    
    group_size = args.group_size
    init_index = args.frame_start
    group_num = int((args.frame_end - args.frame_start) / group_size)

    # base_dir = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/feature_image"
    # save_dir = f"/data/new_disk5/wangph1/output/xyz_smalliter_group40/feature_video"
    base_dir = os.path.join(args.output_path, "feature_image")
    save_dir = os.path.join(args.output_path, "feature_video")
    os.makedirs(save_dir, exist_ok=True)
    # qps = [12, 14, 16, 18, 20, 23, 28]
    # qps = [0, 5, 10, 15, 20, 22, 25]
    # qps = [15]
    qps = [args.qp]
    # qps = [28, 30, 32, 35, 37, 40, 50]
    # qps = [43, 45, 47, 50]
    # qps = [43, 45, 47, 50, 0, 10, 15, 22, 26, 32, 37, 40]
    # qps = [15]
    for qp in qps:
        out_dir = os.path.join(save_dir, "png_all_" + str(qp))
        # out_dir = save_dir
        # if os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        # video_path = os.path.join(out_dir, "video")
        # os.makedirs(video_path, exist_ok=True)

        for group in range(group_num):

            start_index = group * group_size + init_index
            # end_index = (group + 1) * group_size
            group_path = os.path.join(out_dir, "group" + str(group))
            os.makedirs(group_path, exist_ok=True)
            # group_video_path = os.path.join(group_path, "video")
            # os.makedirs(group_video_path, exist_ok=True)
            group_video_path = group_path

            input_group_path = os.path.join(base_dir, "group" + str(group))

            for i in range(0,20):
                if i in [1,3,5]:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp 0 -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
                elif i in [9, 10, 11,  13, 14, 15,  16, 17, 18, 19] and qp > 22:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp 22 -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
                else:
                    os.system(f"ffmpeg -start_number {start_index} -i {input_group_path}/%d_{i}.png -vframes {group_size} -c:v libx264 -qp {qp} -pix_fmt yuvj444p {group_video_path}/{i}.mp4")
                # if i < 3:
                #     os.system(f"ffmpeg -i {video_path}/{i}.mp4 -vf format=gray16le -start_number 0 {out_dir}/%d_{i}.png")
                # else:
                # os.system(f"ffmpeg -i {group_video_path}/{i}.mp4 -vf format=gray -start_number {start_index} {group_path}/%d_{i}.png")
        #get all the file in the dir
        # file_list = os.listdir(save_dir)
        # for file in file_list:
        #     #check if the name of file has 'occupancy'
        #     if 'json' in file or 'atlas' in file or 'occupancy' in file:
        #         os.system(f"cp {save_dir}/{file} {out_dir}/{file}")

    # copy the json
    # shutil.copy(os.path.join(base_dir, "min_max.json"), os.path.join(out_dir, "min_max.json"))
    shutil.copy(os.path.join(base_dir, "viewer_min_max.json"), os.path.join(out_dir, "viewer_min_max.json"))
    shutil.copy(os.path.join(base_dir, "group_info.json"), os.path.join(out_dir, "group_info.json"))


    print("finish")
