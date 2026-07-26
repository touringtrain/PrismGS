#!/usr/bin/env python3
import os
import torch
import imageio
import argparse
from tqdm import tqdm
from scene import Scene, GaussianModel, dataset_readers
from gaussian_renderer import render
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from arguments import ModelParams, PipelineParams
import numpy as np
import time

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def evaluate_model(frame_idx):
    # 记录当前时间
    timestamp = time.time()

    # 将时间戳转换为本地时间
    local_time = time.localtime(timestamp)

    formatted_time = time.strftime("[%d/%m %H:%M:%S]", local_time)
    print(formatted_time)
    # 参数设置
    parser = argparse.ArgumentParser()
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--iteration", type=int, default=-1, help="Model iteration to load")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory")
    args = parser.parse_args()
    args.source_path = os.path.join(args.source_path, f"{frame_idx}")
    args.model_path = os.path.join(args.model_path, f"{frame_idx}")
    # 初始化
    os.makedirs(f"{args.output_dir}/{frame_idx}", exist_ok=True)
    gaussians = GaussianModel(0)
    scene = Scene(model_params.extract(args), shuffle=False, gaussians=gaussians, load_iteration=-1)
    # bg_color = [1, 1, 1] if model_params.extract(args).white_background else [0, 0, 0]
    bg_color = [1, 1, 1]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pipeline_params.extract(args)

    metrics = {"l1": 0, "psnr": 0, "ssim": 0}

    # 遍历所有训练相机
    render_time = 0.0
    cameras = scene.getTrainCameras()
    rendered_cam = 0
    for idx, cam in enumerate(tqdm(cameras, desc="Evaluating")):
        if idx == 23:
            # 渲染图像
            # print(cam.original_image.shape)
            # print(cam.R)
            # print(cam.T)
            # print(cam.FoVx)
            # print(cam.FoVy)
            # measure render time
            rendered_cam += 1
            torch.cuda.synchronize()
            time1 = time.time()
            render_pkg = render(cam, gaussians, pipe, background)
            rendering = render_pkg["render"]  # [3, H, W]
            torch.cuda.synchronize()
            time2 = time.time()
            render_time += (time2 - time1) * 1000

            # 获取真实图像
            gt_image = cam.original_image.to("cuda")  # [3, H, W]

            # 计算指标
            # measure calculate time
            metrics["l1"] += l1_loss(rendering, gt_image).mean().item()
            metrics["psnr"] += psnr(rendering, gt_image).mean().item()
            metrics["ssim"] += ssim(rendering, gt_image).mean().item()

            # 保存渲染结果
            # measure save time
            output_path = os.path.join(args.output_dir, f"{frame_idx}")
            render_np = to8b(rendering.detach().cpu().numpy().transpose(1, 2, 0))
            gt_np = to8b(gt_image.detach().cpu().numpy().transpose(1, 2, 0))
            imageio.imwrite(f"{output_path}/render_{idx:04d}.png", render_np)
            imageio.imwrite(f"{output_path}/gt_{idx:04d}.png", gt_np)

    # 打印平均指标
    num_cams = rendered_cam
    print(f"\nFrame {frame_idx}: Average Metrics (over {num_cams} views):")
    print(f"L1: {metrics['l1']/num_cams:.6f}")
    print(f"PSNR: {metrics['psnr']/num_cams:.2f} dB")
    print(f"SSIM: {metrics['ssim']/num_cams:.4f}")
    print(f"Render Time: {render_time:.4f} ms")
    print(f"Average Render Time per Camera: {render_time / num_cams:.4f} ms")

if __name__ == "__main__":
    start = 100
    end = 101
    for frame_idx in range(start, end, 1):
        # 评估模型
        evaluate_model(frame_idx)