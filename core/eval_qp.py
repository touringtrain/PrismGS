# write a script that run the following commands
# python ../../render_test.py --model_path feature_video/png_all_{0}/qp_{0}/ --source_path ../../dataset/4k_5_umbrella/ --output_dir feature_video/png_all_{0}/qp_{0}/ > feature_video/png_all_{0}/eval.txt
# in which {0} is replaced by qp values in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
import os

qp_values = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
for qp in qp_values:
    model_path = f"sensitivity/png_all_{qp}/qp_{qp}/"
    source_path = "../videogs/VideoGS/dataset/4k_5_umbrella/"
    output_dir = f"sensitivity/png_all_{qp}/qp_{qp}/"
    eval_file = f"sensitivity/png_all_{qp}/eval.txt"
    command = f"python ../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
    os.system(command)
