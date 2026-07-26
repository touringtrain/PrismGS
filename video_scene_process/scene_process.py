# read all images under the same root as the code file

import os
import cv2
import numpy as np

root_dir = os.path.dirname(os.path.abspath(__file__))
image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

images = []
for image_file in image_files:
    image_path = os.path.join(root_dir, image_file)
    image = cv2.imread(image_path)
    images.append(image)
    print(image.shape)
    
# for all images, crop the center 400*850 region and save to 'cropped_images' folder
crop_height = 890
crop_width = 500
cropped_dir = os.path.join(root_dir, 'cropped_images')
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)
for idx, image in enumerate(images):
    height, width, _ = image.shape
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    cropped_image_path = os.path.join(cropped_dir, f'cropped_{image_files[idx]}')
    cv2.imwrite(cropped_image_path, cropped_image)
    print(f'Saved cropped image to {cropped_image_path}')
    
    # for 4k_1: crop a small region at (x: 800-1000, y: 400-600)
    if idx == 0 or idx == 1:
        small_cropped_image = image[450:550, 750:850]
        small_cropped_image_path = os.path.join(cropped_dir, f'small_cropped_{image_files[idx]}')
        cv2.imwrite(small_cropped_image_path, small_cropped_image)
        print(f'Saved small cropped image to {small_cropped_image_path}')
