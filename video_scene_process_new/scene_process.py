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

# sort the images by filename
image_files, images = zip(*sorted(zip(image_files, images)))
    
# for all images, crop the center 400*850 region and save to 'cropped_images' folder
# crop_height = 890
# crop_width = 500
cropped_dir = os.path.join(root_dir, 'cropped_images')

if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)
    
for idx, image in enumerate(images):
    # height, width, _ = image.shape
    # start_x = (width - crop_width) // 2
    # start_y = (height - crop_height) // 2
    # cropped_image = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
    scene_name = image_files[idx].split('_')[0] + '_' + image_files[idx].split('_')[1]  # get scene name like '4k_1'
    cropped_image_path = os.path.join(cropped_dir, scene_name, f'cropped_{image_files[idx]}')
    # cv2.imwrite(cropped_image_path, cropped_image)
    # print(f'Saved cropped image to {cropped_image_path}')
    if not os.path.exists(os.path.join(cropped_dir, scene_name)):
        os.makedirs(os.path.join(cropped_dir, scene_name))
    # for 4k_1: crop a small region at (x: 800-1000, y: 400-600)
    if image_files[idx].startswith('4k_1'):
        large_cropped_image = image[150:1000, 650:1100]
        large_cropped_image_path = os.path.join(cropped_dir, scene_name, f'large_cropped_{image_files[idx]}')
        cv2.imwrite(large_cropped_image_path, large_cropped_image)
        small_cropped_image = image[350:470, 800:950]
        small_cropped_image_path = os.path.join(cropped_dir, scene_name, f'small_cropped_{image_files[idx]}')
        cv2.imwrite(small_cropped_image_path, small_cropped_image)
        print(f'Saved small cropped image to {small_cropped_image_path}')
    elif image_files[idx].startswith('4k_4'):
        if image_files[idx] == "4k_4_3dgstream.png": # resize this image to 1600*1200, only use the resized image, dont save resized image
            resized_image = cv2.resize(image, (1600, 893))
            image = resized_image
        large_cropped_image = image[50:1000, 550:1000]
        large_cropped_image_path = os.path.join(cropped_dir, scene_name, f'large_cropped_{image_files[idx]}')
        cv2.imwrite(large_cropped_image_path, large_cropped_image)
        small_cropped_image = image[50:300, 700:900]
        small_cropped_image_path = os.path.join(cropped_dir, scene_name, f'small_cropped_{image_files[idx]}')
        cv2.imwrite(small_cropped_image_path, small_cropped_image)
        down_cropped_image = image[700:950, 750:950]
        down_cropped_image_path = os.path.join(cropped_dir, scene_name, f'down_cropped_{image_files[idx]}')
        cv2.imwrite(down_cropped_image_path, down_cropped_image)
        # print(f'Saved small cropped image to {small_cropped_image_path}')
    elif image_files[idx].startswith('4k_5'):
        large_cropped_image = image[100:1000, 550:1050]
        large_cropped_image_path = os.path.join(cropped_dir, scene_name, f'large_cropped_{image_files[idx]}')
        cv2.imwrite(large_cropped_image_path, large_cropped_image)
        small_cropped_image = image[300:450, 700:850]
        small_cropped_image_path = os.path.join(cropped_dir, scene_name, f'small_cropped_{image_files[idx]}')
        cv2.imwrite(small_cropped_image_path, small_cropped_image)
        # print(f'Saved small cropped image to {small_cropped_image_path}')
    elif image_files[idx].startswith('3dv_f'):
        if image_files[idx] == "3dv_f_H.png" or image_files[idx] == "3dv_f_3dgstream.png": # resize this image to 1600*1200, only use the resized image, dont save resized image
            resized_image = cv2.resize(image, (1600, 1200))
            image = resized_image
        
        face_cropped_image = image[500:680, 750:930]
        face_cropped_image_path = os.path.join(cropped_dir, scene_name, f'face_cropped_{image_files[idx]}')
        cv2.imwrite(face_cropped_image_path, face_cropped_image)
        dog_cropped_image = image[800:950, 150:300]
        dog_cropped_image_path = os.path.join(cropped_dir, scene_name, f'dog_cropped_{image_files[idx]}')
        cv2.imwrite(dog_cropped_image_path, dog_cropped_image)
        flame_cropped_image = image[920:1120, 670:870]
        flame_cropped_image_path = os.path.join(cropped_dir, scene_name, f'flame_cropped_{image_files[idx]}')
        cv2.imwrite(flame_cropped_image_path, flame_cropped_image)
        hand_cropped_image = image[480:680, 680:880]
        hand_cropped_image_path = os.path.join(cropped_dir, scene_name, f'hand_cropped_{image_files[idx]}')
        cv2.imwrite(hand_cropped_image_path, hand_cropped_image)
        # print(f'Saved cropped image to {face_cropped_image_path}')