# Environment Setup
Install all python dependencies following [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [V^3](https://github.com/AuthorityWang/VideoGS)

# Run AdaGS

## Dataset

We use HiFi4G dataset for 4DGS reconstruction, which consists of 7 dynamic scenes of 200 frames with white background. Each scene is captured by 81 viewpoints under 4k resolution (3840*2160) at 30 FPS. The dataset is too large (more than 200GB) and we include a piece of dataset in folder /4k_4, from frame 100 to frame 105. 

We use Lumos-5G dataset for network testing, which record internet throughput every second. 

## Reconstruct and compress 4DGS scene

We conduct 4DGS reconstruction following V^3 pipeline, the process and generated Gaussian point cloud are not in the scope of AdaGS and are not included. 

## Preprocess

For simplicity, we only include the dancing2 (4k_4) scene in HiFi4G dataset, since the test data of each scene is about 10GB. After 4DGS reconstruction, we compress the generated Gaussian checkpoints to images and store in 4k_4_bandwidth/feature_image.

Then run 'compress_image_2_video_sensitivity' to generate feature video of different qps in /feature_video. After that, run "decompress_video_2_ckpt_sensitivity.py" to decompress the feature videos to anchor checkpoints, and run 'render_sensitivity.py' to get the visual qualities of the anchors. 

## Construct quality model

Run 'quality_model.py', which is the core algorithm of AdaGS, to generate the Pareto frontier based on the tested anchor checkpoints and store in pareto_frontier_dp_4k_4.json. Then run 'decompress_video_2_ckpt_testset.py' to obtain the checkpoints on the Pareto frontier from the feature videos. Use 'render_testset.py' to test their actual visual quality and store in test_results.json. 

## Evaluation

Run '/evaluation/overall_performance.py' to generate the overall performance plots of AdaGS and baselines on Lumos5G throughput trace. For other plots in the paper, please refer to the /plot folder. 

## Visualization

The visualization of the scenes is in all.pdf. 
