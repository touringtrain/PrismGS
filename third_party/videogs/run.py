import os
import argparse
import shutil
import pymeshlab
import open3d as o3d
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default='')
    parser.add_argument('--end', type=int, default='')
    parser.add_argument('--cuda', type=int, default='')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--output', type=str, default='')
    parser.add_argument('--sh', type=str, default='')
    parser.add_argument('--interval', type=str, default='')
    parser.add_argument('--group_size', type=str, default='')
    args = parser.parse_args()

    print(args.start, args.end)

    # os.system("conda activate torch")
    card_id = args.cuda
    data_root_path = args.data
    output_path = args.output
    sh = args.sh
    interval = int(args.interval)
    group_size = int(args.group_size)

    # neus2_meshlab_filter_path = os.path.join(data_root_path, "luoxi_filter.mlx")

    neus2_output_path = os.path.join(output_path, "neus2_output")
    if not os.path.exists(neus2_output_path):
        os.makedirs(neus2_output_path)

    gaussian_output_path = os.path.join(output_path, "checkpoint")

    for i in range(args.start, args.end, group_size * interval):
        group_start = i
        group_end = min(i + group_size * interval, args.end)
        print(group_start, group_end)
        
        frame_path = os.path.join(data_root_path, str(i))
        if not os.path.exists(frame_path):
            os.makedirs(frame_path)
        frame_neus2_output_path = os.path.join(neus2_output_path, str(i))
        if not os.path.exists(frame_neus2_output_path):
            os.makedirs(frame_neus2_output_path)
        frame_neus2_ckpt_output_path = os.path.join(frame_neus2_output_path, "frame.msgpack")
        frame_neus2_mesh_output_path = os.path.join(frame_neus2_output_path, "points3d.obj")
        
        """NeuS2"""
        # neus2 command
        script_path = "scripts/run.py"
        neus2_command = f"cd external/NeuS2 && CUDA_VISIBLE_DEVICES={card_id} python {script_path} --scene {frame_path} --name neus --mode nerf --save_snapshot {frame_neus2_ckpt_output_path} --save_mesh --save_mesh_path {frame_neus2_mesh_output_path} && cd ../.."
        os.system(neus2_command)
        delete_neus2_output_path = os.path.join(frame_path, "output")
        shutil.rmtree(delete_neus2_output_path)

        # revert axis
        mesh1 = o3d.io.read_triangle_mesh(frame_neus2_mesh_output_path)
        vertices = np.asarray(mesh1.vertices)
        vertices = vertices[:,[2,0,1]]
        mesh1.vertices = o3d.utility.Vector3dVector(vertices)
        o3d.io.write_triangle_mesh(frame_neus2_mesh_output_path, mesh1)

        # use pymeshlab to convert obj to point cloud
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(frame_neus2_mesh_output_path)
        # ms.load_filter_script(neus2_meshlab_filter_path)
        # ms.apply_filter_script()
        ms.generate_simplified_point_cloud(samplenum = 100000)
        frame_points3d_output_path = os.path.join(frame_path, "points3d.ply")
        ms.save_current_mesh(frame_points3d_output_path, binary = True, save_vertex_normal = False)


        """ Gaussian """
        # generate output
        frame_model_path = os.path.join(gaussian_output_path, str(i))
        first_frame_iteration = 12000
        first_frame_save_iterations = first_frame_iteration
        first_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python train.py -s {frame_path} -m {frame_model_path} --iterations {first_frame_iteration} --save_iterations {first_frame_save_iterations} --sh_degree {sh} --port 600{card_id}"
        os.system(first_gaussian_command)

        # prune
        prune_iterations = 4000
        prune_gaussian_command = f"CUDA_VISIBLE_DEVICES={card_id} python prune_gaussian.py -s {frame_path} -m {frame_model_path} --sh_degree {sh} --iterations {prune_iterations}"
        os.system(prune_gaussian_command)

        # rest frame
        dynamic_command = f"CUDA_VISIBLE_DEVICES={card_id} python train_dynamic_t.py -s {data_root_path} -m {gaussian_output_path} --sh_degree {sh} --st {group_start} --ed {group_end} --interval {interval}"
        os.system(dynamic_command)

        print(f"Finish {group_start} to {group_end}")