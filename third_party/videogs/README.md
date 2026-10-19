# [SIGGRAPH Asia 2024] V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians
Official implementation for _V^3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians_.

**[Penghao Wang*](https://authoritywang.github.io/), [Zhirui Zhang*](https://github.com/zhangzhr4), [Liao Wang*](https://aoliao12138.github.io/), [Kaixin Yao](https://yaokxx.github.io/), [Siyuan Xie](https://simonxie2004.github.io/about/), [Jingyi Yu†](http://www.yu-jingyi.com/cv/), [Minye Wu†](https://wuminye.github.io/), [Lan Xu†](https://www.xu-lan.com/)**

**SIGGRAPH Asia 2024 (ACM Transactions on Graphics)**

| [Webpage](https://authoritywang.github.io/v3/) | [Paper](https://arxiv.org/pdf/2409.13648) | [Video](https://youtu.be/Z5La9AporRU?si=P95fDRxVYhXZEzYT) | [Training Code](https://github.com/AuthorityWang/VideoGS) | [SIBR Viewer Code](https://github.com/AuthorityWang/VideoGS_SIBR_viewers) | [IOS Viewer Code](https://github.com/zhangzhr4/VideoGS_IOS_viewers) |<br>
![Teaser image](assets/teaser.jpg)

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@article{wang2024v,
  title={V\^{} 3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians},
  author={Wang, Penghao and Zhang, Zhirui and Wang, Liao and Yao, Kaixin and Xie, Siyuan and Yu, Jingyi and Wu, Minye and Xu, Lan},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--13},
  year={2024},
  publisher={ACM New York, NY, USA}
}</code></pre>
  </div>
</section>

## Install
Create a new environment
```
conda create -n videogs python=3.9
conda activate videogs
```
First install CUDA and PyTorch, our code is evaluated on [CUDA 11.6](https://developer.nvidia.com/cuda-11-6-2-download-archive) and [PyTorch 1.13.1+cu116](https://pytorch.org/get-started/previous-versions/#v1131). Then install the following dependencies:
```
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
```

<!-- Install [COLMAP](https://colmap.github.io/install.html) for calibration and undistortion.  -->

Install modified [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) for key frame point cloud generation, please clone it to `external` folder and build it. 
```
cd external
git clone --recursive https://github.com/AuthorityWang/NeuS2_K.git
cd NeuS2_K
cmake . -B build
cmake --build build --config RelWithDebInfo -j
```

## Dataset Preprocess

### Download Dataset

Our code mainly evaluated on multi-view human centric datasets including [ReRF](https://github.com/aoliao12138/ReRF_Dataset), [HiFi4G](https://github.com/moqiyinlun/HiFi4G_Dataset), and [HumanRF](https://synthesiaresearch.github.io/humanrf/#dataset) datasets. Please download the data you needed. 

### Format
Our dataset format is structed as follows:
```
datasets
|   |---xxx (data name)
|   |   |---%d
|   |   |   |---images
|   |   |   |   |---%d.png
|   |   |   |---transforms.json
```
The transforms.json is based on NGP calibration format:
```
{
    "frames": [
        {
            "file_path": "xxx/xxx.png" (file path to the image), 
            "transform_matrix": [
                xxx (extrinsic)
            ], 
            "K": [
                xxx (intrinsic, note can be different for each view)
            ],
            "fl_x": xxx (focal length x),
            "fl_y": xxx (focal length y),
            "cx": xxx (cx),
            "cy": xxx (cx),
            "w": xxx (image width),
            "h": xxx (image height)
        }, 
        {
            ...
        }
    ], 
    "aabb_scale": xxx (aabb scale for NeuS2), 
    "white_transparent": true (if the background is white)
}
```

### HiFi4G Dataset

The dataset is structured as follows:
```
datasets
|---HiFi4G
|   |---xxx (data name)
|   |   |---image_undistortion_white
|   |   |   |---%d            - The frame number, starts from 0.
|   |   |   |   |---%d.png    - Multi-view images, starts from 0.
|   |   |---colmap/sparse/0   - Camera extrinsics and intrinsics in Gaussian Splatting format.
```
Then you need to restruct the dataset and convert colmap calibration to ngp format of transforms.json, simply run the following command:
```
cd preprocess
python hifi4g_process.py --input xxx --output xxx
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for hifi4g_process.py</span></summary>

  #### --input
  Input folder to the original hifi4g dataset
  #### --output
  Output folder to the processed hifi4g dataset
  #### --move
  If move the images to the output folder or copy. True for move, False for copy. 

</details>
<br>

The processed dataset is structured as follows:
```
datasets
|---HiFi4G
|   |---xxx (data name)
|   |   |---%d
|   |   |   |---images
|   |   |   |   |---%d.png
|   |   |   |---transforms.json
```

### ReRF dataset

To process ReRF dataset, you need to re-calibration, undistortion the images and then convert to our format. 

#### Calibration

Install [COLMAP](https://colmap.github.io/install.html) for calibration and undistortion. However, as images without background is hard to calibration, here we provide a colmap calibration for KPOP sequence in ReRF datasets. You can download it from [this link](https://5xmbb1-my.sharepoint.com/:f:/g/personal/auwang_5xmbb1_onmicrosoft_com/Ek6nsqEIzFxAi7j6H2FKv8UB6lNV0_h_JcLuv7JwG7ZLTg?e=SclyNr). If you need other sequence's calibration for ReRF dataset, please contact by email [wangph1@shanghaitech.edu.cn](wangph1@shanghaitech.edu.cn)

#### Undistortion

With installed colmap and colmap calibration, you can undistortion the other frames by the command
```
cd preprocess
python undistortion.py --input xxx --output xxx --calib xxx(the path to colmap calibration) --start xxx(start frame) --end xxx(end frame)
```

Then follow the code in undistortion.py, undistortion the calibration, and use colmap2k.py to generate the transform.json file. 

Finally, the processed dataset is structured as follows:
```
datasets
|---ReRF
|   |---xxx (data name)
|   |   |---%d
|   |   |   |---images (undistorted images)
|   |   |   |   |---%d.png
|   |   |   |---transforms.json
```

## Train

For processed data, lanuch training with `train_sequence.py`
```
python train_sequence.py --start 0 --end 200 --cuda 0 --data datasets/HiFi4G/0932dancer3 --output output/0923dancer3 --sh 0 --interval 1 --group_size 20 --resolution 2
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train_sequence.py</span></summary>

  #### --start
  The frame id to start training
  #### --end
  The frame id to end training
  #### --cuda
  The CUDA device for training
  #### --data
  The path to the dataset, note that this should be the folder containing frames from start to end
  #### --output
  The output path for trained frame
  #### --sh
  Order of spherical harmonics to be used. ```0``` by default.
  #### --interval 1
  The interval between frames. For example, if set to 2, the training frames will be 0, 2, 4, 6, ...
  #### --group_size 20
  The number of frames to trained in a group
  #### --resolution 2
  Specifies resolution of the loaded images before training. If provided ```1, 2, 4``` or ```8```, uses original, 1/2, 1/4 or 1/8 resolution, respectively. For all other values, rescales the width to the given number while maintaining image aspect. **If not set and input image width exceeds 1.6K pixels, inputs are automatically rescaled to this target.**


</details>
<br>

After training, the checkpoints in the output folder is structured as follows:
```
output
|---0923dancer3
|   |---checkpoint
|   |   |---%d (each frame ckpt folder)
|   |   |---record (record config and training file)
|   |---neus2_output
```

## Compress

After getting the Gaussian point clouds, we can compress them by the following command:
```
python compress_ckpt_2_image_precompute.py --frame_start 100 --frame_end 140 --group_size 20 --interval 1 --ply_path ~/workspace/output/v3/0923dancer3/checkpoint/ --output_folder ~/workspace/output/v3/0923dancer3/feature_image --sh_degree 0
```
The frame trained is [100, 140), so is 40 frames. 
The output structure will be: 
```
output
|---0923dancer3
|   |---checkpoint
|   |---feature_image
|   |   |---group%d (each group's images)
|   |   |---min_max.json (store the min max value for each frame)
|   |   |---viewer_min_max.json (same as min_max.json, different struct)
|   |   |---group_info.json (store the each group frame index)
|   |---neus2_output
```

Then compress images to video by the following command:
```
python compress_image_2_video.py --frame_start 100 --frame_end 140 --group_size 20 --output_path ~/workspace/output/v3/0923dancer3 --qp 25
```
The qp value is the parameter for compression, lower refers to higher quality, but larger size.

The output structure will be: 
```
output
|---0923dancer3
|   |---checkpoint
|   |---feature_image
|   |---feature_video
|   |   |---group%d (each group's videos)
|   |   |   |---%d.mp4 (each attribute's video)
|   |   |---viewer_min_max.json (store each frame min max info)
|   |   |---group_info.json (store the each group frame index)
|   |---neus2_output
```

Note that the `compress_image_2_video.py` need to be executed on linux OS due to video codec. 

Finally, the compressed video folder can be hosted by nginx server and use our [volumetric video viewer]() to play. 

## Acknowledgement
Our code is based on original [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation. We also refer [NeuS2](https://vcai.mpi-inf.mpg.de/projects/NeuS2/) for fast key frame point cloud generation, and [3DGStream](https://sjojok.top/3dgstream/) for the inspiration of fast training strategy.

Thanks for [Zhehao Shen](https://github.com/moqiyinlun) for his help on datasets process. 

If you find our work useful in your research, please consider citing our paper.
```
@article{wang2024v,
  title={V\^{} 3: Viewing Volumetric Videos on Mobiles via Streamable 2D Dynamic Gaussians},
  author={Wang, Penghao and Zhang, Zhirui and Wang, Liao and Yao, Kaixin and Xie, Siyuan and Yu, Jingyi and Wu, Minye and Xu, Lan},
  journal={ACM Transactions on Graphics (TOG)},
  volume={43},
  number={6},
  pages={1--13},
  year={2024},
  publisher={ACM New York, NY, USA}
}
```