# TensoRF-Paddle
A PaddlePaddle Implementation for [TensoRF: Tensorial Radiance Fields](https://arxiv.org/abs/2203.09517). This work present a novel 
approach to model and reconstruct radiance fields, which achieves super
**fast** training process, **compact** memory footprint and **state-of-the-art** rendering quality.<br><br>

https://user-images.githubusercontent.com/16453770/158920837-3fafaa17-6ed9-4414-a0b1-a80dc9e10301.mp4

## Installation

- [PaddlePaddle Installation](https://www.paddlepaddle.org.cn/install/quick)
    - Versions：PaddlePaddle develop, Python>=3.7

- TensoRF-Paddle Installation, use the following command:
```
git clone https://github.com/kongdebug/TensoRF-Paddle
cd TensoRF-Paddle
pip install -r requirements.txt
```


## Dataset
* [Synthetic-NeRF](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1), this dataset has been uploaded to [AI Studio](https://aistudio.baidu.com/aistudio/datasetdetail/136816). The prepared data organization is as follows:

```
TensoRF-Paddle/data
└── nerf_synthetic
    └── lego
```

## Quick Start
The training script is in `train.py`, to train a TensoRF:

```
python train.py --config configs/lego.txt
```

we provide a few examples in the configuration folder, please note:

 `dataset_name`, choices = ['blender'];

 `shadingMode`, choices = ['MLP_Fea', 'SH'];

 `model_name`, choices = ['TensorVMSplit', 'TensorCP'], corresponding to the VM and CP decomposition, . 
 You need to uncomment the last a few rows of the configuration file if you want to training with the TensorCP model；

 `n_lamb_sigma` and `n_lamb_sh` are string type refer to the basis number of density and appearance along XYZ
dimension;

 `N_voxel_init` and `N_voxel_final` control the resolution of matrix and vector;

 `N_vis` and `vis_every` control the visualization during training;

  You need to set `--render_test 1`/`--render_path 1` if you want to render testing views or path after training. 

More options refer to the `opt.py`. 

## Reproducibility:

We trained the TensoRF-VM-192 model using only the data from the 'lego' scenario, with a training iteration count of 30k, a time of about 30 minutes, and a rendering time of about 8 minutes. The recurrence result is as follows: 

| Model | Sence | PSNR | SSIM | LPIPS-vgg | LPIPS-alex |
| --- | --- | --- | --- | --- | --- |
| TensoRF-VM-192 | LEGO | 36.60 | 0.9816 | 0.020 | 0.008 |

The trained weight can be obtained through the following link, and the extraction code is: yivv

[https://pan.baidu.com/s/1WWjKfvlLA7GEI3hflZo_8g ](https://pan.baidu.com/s/1WWjKfvlLA7GEI3hflZo_8g)



## Rendering

```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --render_only 1 --render_test 1 
```

You can just simply pass `--render_only 1` and `--ckpt path/to/your/checkpoint` to render images from a pre-trained
checkpoint. You may also need to specify what you want to render, like `--render_test 1`, `--render_train 1` or `--render_path 1`.
The rendering results are located in your checkpoint folder. 

## Extracting mesh
You can also export the mesh by passing `--export_mesh 1`:
```
python train.py --config configs/lego.txt --ckpt path/to/your/checkpoint --export_mesh 1
```
Note: After exporting the mesh, the training will restart. If you don't need it, you can exit directly.
    

## Acknowledge
Thanks Chen et al. for opening source [TensoRF](https://github.com/apchenstu/TensoRF), which has helped build the Paddle version of TensoRF.

## TODO

The TensoRF model is being incorporated into the official Paddle Rendering rendering suite, but there are still some bugs that need to be resolved.
