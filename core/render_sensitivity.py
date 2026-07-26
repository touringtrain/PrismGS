# for ablation/position_low_{0, 2, 4}_qp{10, 20, 30, 40, 50}, execute render_test.py


import os

# stem_dir = "sensitivity/4k_7/checkpoints"
# dataset_path = "../videogs/VideoGS/dataset/4k_7_nunchaku/"

stem_dir = "4k_4_dancing"
dataset_path = "../videogs/VideoGS/dataset/4k_4_dancing/"

position_high_values = [1, 3, 5]
position_low_values = [0, 2, 4]
color_values = [9, 10, 11]
opacity_values = [12]
scale_values = [13, 14, 15]
rotation_values = [16, 17, 18, 19]

# qp_values = [0, 10, 20, 30, 40, 50]
qp_values = [20]

# for qp in qp_values:
#     model_path = f"{stem_dir}/position_high_qp{qp}/"
#     source_path = dataset_path
#     output_dir = f"{stem_dir}/position_high_qp{qp}/"
#     eval_file = f"{stem_dir}/position_high_qp{qp}/eval.txt"
#     command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#     os.system(command)

# for position_low in position_low_values:
#     for qp in qp_values:
#         model_path = f"sensitivity/checkpoints/position_low_{position_low}_qp{qp}/"
#         source_path = "../videogs/VideoGS/dataset/4k_5_umbrella/"
#         output_dir = f"sensitivity/checkpoints/position_low_{position_low}_qp{qp}/"
#         eval_file = f"sensitivity/checkpoints/position_low_{position_low}_qp{qp}/eval.txt"
#         command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#         os.system(command)

for qp in qp_values:
    model_path = f"{stem_dir}/position_low_qp{qp}/"
    source_path = dataset_path
    output_dir = f"{stem_dir}/position_low_qp{qp}/"
    eval_file = f"{stem_dir}/position_low_qp{qp}/eval.txt"
    command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
    os.system(command)

# for color in color_values:
#     for qp in qp_values:
#         model_path = f"sensitivity/checkpoints/color_{color}_qp{qp}/"
#         source_path = "../videogs/VideoGS/dataset/4k_5_umbrella/"
#         output_dir = f"sensitivity/checkpoints/color_{color}_qp{qp}/"
#         eval_file = f"sensitivity/checkpoints/color_{color}_qp{qp}/eval.txt"
#         command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#         os.system(command)
        
# for qp in qp_values:
#     model_path = f"{stem_dir}/color_qp{qp}/"
#     source_path = dataset_path
#     output_dir = f"{stem_dir}/color_qp{qp}/"
#     eval_file = f"{stem_dir}/color_qp{qp}/eval.txt"
#     command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#     os.system(command)
        
# for qp in qp_values:
#     model_path = f"{stem_dir}/opacity_qp{qp}/"
#     source_path = dataset_path
#     output_dir = f"{stem_dir}/opacity_qp{qp}/"
#     eval_file = f"{stem_dir}/opacity_qp{qp}/eval.txt"
#     command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#     os.system(command)
        
# for scale in scale_values:
#     for qp in qp_values:
#         model_path = f"sensitivity/checkpoints/scale_{scale}_qp{qp}/"
#         source_path = "../videogs/VideoGS/dataset/4k_5_umbrella/"
#         output_dir = f"sensitivity/checkpoints/scale_{scale}_qp{qp}/"
#         eval_file = f"sensitivity/checkpoints/scale_{scale}_qp{qp}/eval.txt"
#         command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#         os.system(command)
        
# for qp in qp_values:
#     model_path = f"{stem_dir}/scale_qp{qp}/"
#     source_path = dataset_path
#     output_dir = f"{stem_dir}/scale_qp{qp}/"
#     eval_file = f"{stem_dir}/scale_qp{qp}/eval.txt"
#     command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#     os.system(command)
        
# for rotation in rotation_values:
#     for qp in qp_values:
#         model_path = f"sensitivity/checkpoints/rotation_{rotation}_qp{qp}/"
#         source_path = "../videogs/VideoGS/dataset/4k_5_umbrella/"
#         output_dir = f"sensitivity/checkpoints/rotation_{rotation}_qp{qp}/"
#         eval_file = f"sensitivity/checkpoints/rotation_{rotation}_qp{qp}/eval.txt"
#         command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#         os.system(command)


# for qp in qp_values:
#     model_path = f"{stem_dir}/rotation_qp{qp}/"
#     source_path = dataset_path
#     output_dir = f"{stem_dir}/rotation_qp{qp}/"
#     eval_file = f"{stem_dir}/rotation_qp{qp}/eval.txt"
#     command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
#     os.system(command)
