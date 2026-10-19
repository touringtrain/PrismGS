import os
import argparse

def undistortion(image_path, input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    image_undistorter = "colmap image_undistorter --image_path {} --input_path {} --output_path {} --output_type COLMAP".format(image_path,input_path,output_path)
    os.system(image_undistorter)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",type = str)
    parser.add_argument("--output",type = str)
    parser.add_argument("--calib", type = str)
    parser.add_argument("--start", type = str)
    parser.add_argument("--end", type = str)
    parser.add_argument("--interval", type = str, default="1")
    parser=parser.parse_args()
    start_frame = int(parser.start)
    end_frame = int(parser.end)
    interval = int(parser.interval)

    input_path = parser.input
    output_path = parser.output
    calib_path = parser.calib
    for frame in range(start_frame, end_frame, interval):
        print("Processing frame: ", frame)
        input_image_path = os.path.join(input_path, str(frame), "images")
        if not os.path.exists(input_image_path):
            raise Exception(f"Path {input_image_path} does not exist.")
        
        output_image_path = os.path.join(output_path, str(frame), "image_undistortion_white")
        undistortion(input_image_path, calib_path, output_image_path)