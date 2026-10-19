#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import DynamicScene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.system_utils import searchForMaxIteration

import numpy as np
from plyfile import PlyData

def finetune(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, last_ckpt_path, last_ckpt_iter):
    first_iter = 0
    gaussians = GaussianModel(0)
    scene = DynamicScene(dataset)
    gaussians.load_ply(last_ckpt_path)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(None, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, gaussians, scene, render, (pipe, background))

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    with torch.no_grad():
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        save_pcd_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(last_ckpt_iter + opt.iterations))
        gaussians.save_ply(os.path.join(save_pcd_path, "point_cloud.ply"))

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, gaussians, scene : DynamicScene, renderFunc, renderArgs):

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
        torch.cuda.empty_cache()

def get_ply_matrix(file_path):
    plydata = PlyData.read(file_path)
    num_vertices = len(plydata['vertex'])
    num_attributes = len(plydata['vertex'].properties)
    data_matrix = np.zeros((num_vertices, num_attributes), dtype=float)
    for i, name in enumerate(plydata['vertex'].data.dtype.names):
        data_matrix[:, i] = plydata['vertex'].data[name]
    return data_matrix

def get_attribute(sh_degree):
    frest_dim = 3 * (sh_degree + 1) * (sh_degree + 1) - 3
    attribute_names = []
    attribute_names.append('x')
    attribute_names.append('y')
    attribute_names.append('z')
    attribute_names.append('nx')
    attribute_names.append('ny')
    attribute_names.append('nz')
    for i in range(3):
        attribute_names.append('f_dc_' + str(i))
    for i in range(frest_dim):
        attribute_names.append('f_rest_' + str(i))
    attribute_names.append('opacity')
    for i in range(3):
        attribute_names.append('scale_' + str(i))
    for i in range(4):
        attribute_names.append('rot_' + str(i))

    return attribute_names

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # prune percentage
    prune_percentage = 0.5 # 20%
    last_ckpt_iter = 12000
    # search for the last checkpoint
    pcd_path = os.path.join(args.model_path, "point_cloud")
    last_ckpt_path = os.path.join(pcd_path, "iteration_{}".format(last_ckpt_iter), "point_cloud.ply")

    sh_degree = 0

    pcd = get_ply_matrix(last_ckpt_path)
    print("Loaded point cloud with shape: ", pcd.shape)
    num_points = pcd.shape[0]
    num_points_to_prune = int(num_points * prune_percentage)
    # sort by opacity
    # opacity is the -8th column
    sorted_indices = np.argsort(pcd[:, -8])
    # prune the first num_points_to_prune points
    pruned_pcd = pcd[sorted_indices[num_points_to_prune:]]
    pruned_num_points = pruned_pcd.shape[0]
    print("Pruned point cloud with shape: ", pruned_pcd.shape)

    # save the pruned pcd
    pruned_pcd_path = last_ckpt_path.replace(".ply", "_pruned.ply")
    attribute_list = get_attribute(sh_degree)

    # write the new ply file
    with open(os.path.join(pruned_pcd_path), 'wb') as ply_file:
        ply_file.write(b"ply\n")
        ply_file.write(b"format binary_little_endian 1.0\n")
        ply_file.write(b"element vertex %d\n" % pruned_num_points)
        
        for attribute_name in attribute_list:
            ply_file.write(b"property float %s\n" % attribute_name.encode())
        
        ply_file.write(b"end_header\n")
        
        for i in range(pruned_num_points):
            vertex_data = pruned_pcd[i].astype(np.float32).tobytes()
            ply_file.write(vertex_data)

    finetune(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, pruned_pcd_path, last_ckpt_iter)

    # All done
    print("\nTraining complete.")
