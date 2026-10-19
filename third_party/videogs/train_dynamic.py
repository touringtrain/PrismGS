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
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DynamicScene, GTPTGaussianModel
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import shutil
import copy
import torch.nn.functional as F
import plyfile
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def entropy_regularization_loss(current_frame_gaussian, last_frame_gaussian):
    # current_frest = current_frame_gaussian._features_rest
    current_scale = current_frame_gaussian._scaling
    current_rotation = current_frame_gaussian._rotation
    current_opacity = current_frame_gaussian._opacity
    current_attribute = []
    for i in range(current_scale.shape[1]):
        current_attribute.append(current_scale[:, i])
    for i in range(current_rotation.shape[1]):
        current_attribute.append(current_rotation[:, i])
    for i in range(current_opacity.shape[1]):
        current_attribute.append(current_opacity[:, i])

    last_scale = last_frame_gaussian._scaling
    last_rotation = last_frame_gaussian._rotation
    last_opacity = last_frame_gaussian._opacity
    last_attribute = []
    for i in range(last_scale.shape[1]):
        last_attribute.append(last_scale[:, i])
    for i in range(last_rotation.shape[1]):
        last_attribute.append(last_rotation[:, i])
    for i in range(last_opacity.shape[1]):
        last_attribute.append(last_opacity[:, i])

    quantization_range = 255
    loss = 0.0
    for idx in range(len(current_attribute)):
        delta_attribute = current_attribute[idx] - last_attribute[idx]
        # if zero return
        if torch.sum(delta_attribute) == 0:
            return 0.0
        # delta_attribute_normalize = (delta_attribute - torch.min(delta_attribute)) / (torch.max(delta_attribute) - torch.min(delta_attribute))
        delta_attribute_min = torch.min(delta_attribute)
        delta_attribute_max = torch.max(delta_attribute)
        delta_attribute_normalize = (delta_attribute - delta_attribute_min) / (delta_attribute_max - delta_attribute_min) * quantization_range
        # generate -1/2 to 1/2 noise
        disturb_noise = np.random.uniform(-0.5, 0.5)
        disturb_delta_attribute = delta_attribute_normalize + disturb_noise
        disturb_delta_attribute_up = disturb_delta_attribute + 0.5
        disturb_delta_attribute_down = disturb_delta_attribute - 0.5

        m1 = torch.distributions.normal.Normal(torch.mean(disturb_delta_attribute_up), torch.std(disturb_delta_attribute_up))
        m2 = torch.distributions.normal.Normal(torch.mean(disturb_delta_attribute_down), torch.std(disturb_delta_attribute_down))

        cdf1 = m1.cdf(disturb_delta_attribute)
        cdf2 = m2.cdf(disturb_delta_attribute)

        cdf_diff = cdf1 - cdf2
        loss += -torch.log2(torch.abs(cdf_diff).sum()) / current_attribute[idx].shape[0]

    return loss

def temporal_loss(current_frame_gaussian, last_frame_gaussian):
    # current_frest = current_frame_gaussian._features_rest
    # current_xyz = current_frame_gaussian._xyz
    # current_fdc = current_frame_gaussian._features_dc
    current_scale = current_frame_gaussian._scaling
    current_rotation = current_frame_gaussian._rotation
    current_opacity = current_frame_gaussian._opacity

    current_attribute = [current_scale, current_rotation, current_opacity]

    # last_frest = last_frame_gaussian._features_rest
    # last_xyz = last_frame_gaussian._xyz
    # last_fdc = last_frame_gaussian._features_dc
    last_scale = last_frame_gaussian._scaling
    last_rotation = last_frame_gaussian._rotation
    last_opacity = last_frame_gaussian._opacity

    last_attribute = [last_scale, last_rotation, last_opacity]

    loss = 0.0
    for idx in range(len(current_attribute)):
        att_loss = l2_loss(current_attribute[idx], last_attribute[idx])
        loss += att_loss

    return loss

def train_rt_network(dataset, scene, pipe, last_model_path, init_model_path, gtp_iter, load_last_rt_model, load_init_rt_model):
    first_iter = 0
    gaussians = GTPTGaussianModel(dataset.sh_degree)
    # find the last checkpoint
    last_pcd_iter = searchForMaxIteration(os.path.join(last_model_path, "point_cloud"))
    last_pcd_path = os.path.join(last_model_path, "point_cloud", "iteration_" + str(last_pcd_iter), "point_cloud.ply")
    print("Loading last pcd model from: ", last_pcd_path)
    gaussians.load_ply(last_pcd_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    lambda_dssim = 0.2

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, gtp_iter), desc="Training T progress")
    first_iter += 1
    for iteration in range(first_iter, gtp_iter + 1):        

        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        bg = background

        # first predict gtp
        gaussians.global_predict()
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            if iteration == gtp_iter:
                progress_bar.close()
            if iteration % 10 == 0:
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            # Optimizer step
            if iteration < gtp_iter:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    with torch.no_grad():
        print("\n[ITER {}] Saving GTP Gaussians".format(iteration))
        save_gtp_pcd_path = os.path.join(dataset.model_path, "gtp_pcd/iteration_{}".format(iteration))
        gaussians.save_ply(os.path.join(save_gtp_pcd_path, "point_cloud.ply"))
        print("\n[ITER {}] Saving GTP Checkpoint".format(iteration))
        # save_gtp_ckpt_path = os.path.join(dataset.model_path, "gtp_ckpt/iteration_{}".format(iteration))
        # os.makedirs(save_gtp_ckpt_path, exist_ok = True)
        # gaussians.rt_model.dump_ckpt(os.path.join(save_gtp_ckpt_path, "rt_ckpt.pth"))

def finetune(dataset, scene, opt, pipe, last_model_path, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)


    # load gtp pcd and finetune
    last_model_iter = searchForMaxIteration(os.path.join(dataset.model_path, "gtp_pcd"))
    print("Loading last gtp model from: ", os.path.join(dataset.model_path, "gtp_pcd", "iteration_" + str(last_model_iter)))
    gaussians.load_ply(os.path.join(dataset.model_path, "gtp_pcd", "iteration_" + str(last_model_iter), "point_cloud.ply"))
    gaussians.training_setup(opt)

    last_gaussians = GaussianModel(dataset.sh_degree)
    last_model_iter = searchForMaxIteration(os.path.join(last_model_path, "point_cloud"))
    print("Temporal loss and entropy Loading last pcd model from: ", os.path.join(last_model_path, "point_cloud", "iteration_" + str(last_model_iter), "point_cloud.ply"))
    last_gaussians.load_ply(os.path.join(last_model_path, "point_cloud", "iteration_" + str(last_model_iter), "point_cloud.ply"))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Finetune progress")
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
        temporal_loss_value = temporal_loss(gaussians, last_gaussians)
        entropy_loss = entropy_regularization_loss(gaussians, last_gaussians)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + opt.lambda_temporal * temporal_loss_value + opt.lambda_entropy * entropy_loss
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
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), gaussians)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

    # always save gaussian after finetune
    with torch.no_grad():
        print("\n[ITER {}] Saving Gaussians".format(iteration))
        save_pcd_path = os.path.join(dataset.model_path, "point_cloud/iteration_{}".format(iteration))
        gaussians.save_ply(os.path.join(save_pcd_path, "point_cloud.ply"))

def dynamic_training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, start_frame, end_frame, interval_frame):

    if not os.path.exists(dataset.model_path):
        os.makedirs(dataset.model_path)

    print("Using keyframe {}".format(start_frame))
    # test if keyframe is in the dataset
    frame_list = os.listdir(dataset.model_path)
    if str(start_frame) not in frame_list:
        print("Keyframe model not find")
        return
    
    gtp_iterations = 800
    # gtp_iterations = 4000
    # finetune_iterations = 3500
    finetune_iterations = 2000
    load_last_rt_model = False
    load_init_rt_model = False

    testing_iterations = [finetune_iterations]

    # record code
    os.makedirs(os.path.join(dataset.model_path, "record"), exist_ok = True)
    shutil.copy(__file__, os.path.join(dataset.model_path, "record", "train_dynamic.py"))
    shutil.copy("config/config_hash.json", os.path.join(dataset.model_path, "record", "config_hash.json"))
    shutil.copy("scene/gaussian_model.py", os.path.join(dataset.model_path, "record", "gaussian_model.py"))
    shutil.copy("scene/global_t_field.py", os.path.join(dataset.model_path, "record", "global_rt_field.py"))

    init_model_path = os.path.join(dataset.model_path, str(0))

    for frame in range(start_frame + 1, end_frame + 1, interval_frame):
        print("Training frame {}".format(frame))
        # ready dataset and opt into frame
        frame_dataset = copy.copy(dataset)
        frame_dataset.model_path = os.path.join(dataset.model_path, str(frame))
        frame_dataset.source_path = os.path.join(dataset.source_path, str(frame))

        frame_opt = copy.copy(opt)
        frame_opt.iterations = finetune_iterations

        # ready scene
        tb_writer = prepare_output_and_logger(frame_dataset)
        scene = DynamicScene(frame_dataset)
        
        # learn from last frame
        last_frame = frame - interval_frame
        last_model_path = os.path.join(dataset.model_path, str(last_frame))

        # train rt for the frame using the keyframe model
        train_rt_network(frame_dataset, scene, pipe, last_model_path, init_model_path, gtp_iterations, load_last_rt_model, load_init_rt_model)

        # finetune wrapped model
        finetune(frame_dataset, scene, frame_opt, pipe, last_model_path, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)

        # clean up
        del scene
        del tb_writer

        print("Training frame {} done".format(frame))
        

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, gaussians):
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_500])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_500])
    parser.add_argument("--st", type=int, default=0)
    parser.add_argument("--ed", type=int, default=0)
    parser.add_argument("--interval", type=int, default=0)

    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    print(f"train with keyframe {args.st}")
    print(f"train from frame {args.st + args.interval} to frame {args.ed}")
    dynamic_training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.st, args.ed, args.interval)

    # All done
    print("\nTraining complete.")
