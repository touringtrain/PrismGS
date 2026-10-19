import os
import argparse
import natsort
from tqdm import tqdm
import shutil

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process HIFI4G data')
    parser.add_argument('--input', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--output', type=str, required=True, help='Path to the output data directory')
    parser.add_argument('--move', type=bool, default=False, help='If move the original data to the target folder')
    args = parser.parse_args()

    # assert input and output are same
    assert args.input != args.output, 'Input and output directories are same'

    if not os.path.exists(args.input):
        raise ValueError('Input directory does not exist')
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # generate transforms.json
    text_path = os.path.join(args.input, 'colmap', 'sparse')
    output_json_path = os.path.join(args.output, 'transforms.json')
    colmap2k_cmd = f"python colmap2k.py --text {text_path} --out {output_json_path} --keep_colmap_coords"
    os.system(colmap2k_cmd)

    # move the data
    images_folder_path = os.path.join(args.input, 'image_undistortion_white')
    frames = os.listdir(images_folder_path)
    frames = natsort.natsorted(frames)

    for frame in tqdm(frames):
        frame_source_path = os.path.join(images_folder_path, frame)
        # frame_source_path = os.path.join(images_folder_path, frame, 'image_undistortion_white', 'images')
        frame_target_path = os.path.join(args.output, frame, 'images')
        if args.move:
            shutil.move(frame_source_path, frame_target_path)
        else:
            shutil.copytree(frame_source_path, frame_target_path)
    
        # copy json
        shutil.copy(output_json_path, os.path.join(args.output, frame, 'transforms.json'))