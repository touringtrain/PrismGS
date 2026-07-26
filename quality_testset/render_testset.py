import os
import json

stem_dir = "checkpoints"
dataset_path = "../../videogs/VideoGS/dataset/4k_5_umbrella/"

# check all dirs starting with "test" in stem_dir

test_qp_values = []
size_list = []

# empty the test_results.json file
with open("test_results.json", "w") as f:
    f.write("[\n")

with open("../pareto_frontier_dp.json", "r") as f:
    pareto_data = json.load(f)
    for item in pareto_data:
        config = item["config"]
        qp_list = [
            config["position_low"],
            config["color"],
            config["opacity"],
            config["scale"],
            config["rotation"]
        ]
        test_qp_values.append(qp_list)
        size_list.append(item["bitrate_kb"])
    
for test_qp, size in zip(test_qp_values, size_list):
    dir_name = f"test_{test_qp[0]}_{test_qp[1]}_{test_qp[2]}_{test_qp[3]}_{test_qp[4]}"
    model_path = f"{stem_dir}/{dir_name}/"
    source_path = dataset_path
    output_dir = f"{stem_dir}/{dir_name}/"
    eval_file = f"{stem_dir}/{dir_name}/eval.txt"
    command = f"python ../../videogs/VideoGS/render_test.py --model_path {model_path} --source_path {source_path} --output_dir {output_dir} > {eval_file}"
    os.system(command)
    
    # read the eval.txt to get the psnr and ssim
    psnr = 0
    ssim = 0
    with open(eval_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "PSNR" in line:
                psnr = line.split(":")[-1].strip()
            if "SSIM" in line:
                ssim = line.split(":")[-1].strip()
    print(f"Results for {dir_name}: PSNR = {psnr}, SSIM = {ssim}")

    # save the results to a json file, with indentation
    results = {
        "position_low": test_qp[0],
        "color": test_qp[1],
        "opacity": test_qp[2],
        "scale": test_qp[3],
        "rotation": test_qp[4],
        "bitrate_kb": size,
        "psnr": float(psnr.split()[0]),
        "ssim": float(ssim)
    }
    
    with open("test_results.json", "a") as f:
        json.dump(results, f, indent=2)
        f.write(",\n")  # 添加逗号和换行

with open("test_results.json", "a") as f:
    f.write("\n]")