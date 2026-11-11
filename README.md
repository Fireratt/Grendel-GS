<p align="center">
    <!-- license badge -->
    <a href="https://github.com/nerfstudio-project/nerfstudio/blob/master/LICENSE">
        <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"></a>
    <!-- stars badge -->
    <a href="https://github.com/nyu-systems/Grendel-GS/stargazers">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/nyu-systems/Grendel-GS?style=social"/>
    </a>
    <!-- community badges -->
<!--     <a href="https://discord.gg/uMbNqcraFc"><img src="https://dcbadge.vercel.app/api/server/uMbNqcraFc?style=plastic"/></a> -->
    <!-- last commit badge -->
<!--     <a href="https://github.com/TarzanZhao/Dist-GS/commits/master">
        <img alt="Last Commit" src="https://img.shields.io/github/last-commit/TarzanZhao/Dist-GS"/>
    </a> -->
    <!-- pull requests badge -->
    <a href="https://github.com/nyu-systems/Grendel-GS/pulls">
        <img alt="Pull Requests" src="https://img.shields.io/github/issues-pr/nyu-systems/Grendel-GS"/>
    </a>

</p>


<div align="center">

Grendel-GS
===========================
_<h4>Gaussian Splatting at Scale with Distributed Training System</h4>_

### [Paper](https://arxiv.org/abs/2406.18533) | [Project Page](https://daohanlu.github.io/scaling-up-3dgs/)

<div align="left">

<div align="center">
    <img src="assets/teaser.png" width="900">
</div>

<details>
  <summary> <strong> Click Here to Download Pre-trained Models behind the above visualizations </strong> </summary>
  
  - [Pre-trained Rubble Model (On the left)](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr273-rubble-pointcloud.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20250924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250924T183701Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=085ded68e80e48af7d8213a9b80f2e91eef31601df80ce493a8f9ffa3d432266), [Corresponding Evaluation Images](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr273-rubble-image.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20250924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250924T183741Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=cc5faa5e5a35203600b5707b455b840a4b7349883413f31cfd0c7ef01b530d1a)
  - [Pre-trained MatrixCity Model (On the right)](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr270-matrixcity-blockall-pointcloud.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20250924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250924T183718Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=66c063fa5c5424c931df7e52d596d64ee7272a4b5c2c64634da64fa010f1f38c), [Corresponding Evaluation Images](https://3dgs-public.s3.amazonaws.com/Grendal-GS-checkpoints-and-evalutation-images/psnr270-matrixcity-blockall-images.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAXBPATLFWQSADNSG5%2F20250924%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250924T183755Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=e39e6efdbd5f259d1def50fe0c069d6f693d0558a76059d374126fd736fa6003)

</details>

<!-- 
### [Rubble Model ](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images](https://www.wpeebles.com/DiT) | [MatrixCity Model](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 

### [Pre-trained Rubble Model(the left one)](https://www.wpeebles.com/DiT) | [Rubble Evaluation Images(the left one)](https://www.wpeebles.com/DiT) | [Pre-trained MatrixCity Model(the right one)](https://www.wpeebles.com/DiT) |  [MatrixCity Evaluation Images](https://www.wpeebles.com/DiT) 
-->



# Overview

We design and implement **Grendel-GS**, which serves as a distributed implementation of 3D Gaussian Splatting training. We aim to help 3DGS achieve its *scaling laws* with distributed system support, just as the achievements of current LLMs rely on distributed system. 

By using Grendel, your 3DGS training could leverage multiple GPUs' capability to achieve significantly ***faster training***, supports a substantially ***more Gaussians*** in GPU memory, and ultimately allows for the reconstruction of ***larger-area***, ***higher-resolution*** scenes to better PSNR. Grendel-GS retains the original algorithm, making it a ***direct and safe replacement*** for original 3DGS implementation in any Gaussian Splatting workflow or application.

For examples, with 4 GPU, Grendel-GS allows you to:
- Train Mip360 >3.5 times faster.
- Support directly training large-scale 4K scenes(Mega-NeRF Rubble) using >40 millions gaussians without OOM.
- Train the Temple\&Tanks Truck scene to PSNR 23.79 within merely 45 seconds (on 7000 images)


<!-- 
*Many more new features are developing now, following us!*
-->
> ### 📢 News 
> - 8.3.2024 - gsplat has merged our multi-GPU training strategy, see [PR253](https://github.com/nerfstudio-project/gsplat/pull/253)!
> - 7.15.2024 - We now support gsplat as the CUDA backend during training!

> ### 🌟 Follow us for future updates! Interested in collaborating or contributing? [**Email us!**](mailto:hz3496@nyu.edu)

**Table of contents**
-----
- [Why use Grendel-GS](#why-use-Grendel-gs)
- [How to use Grendel-GS](#how-to-use-Grendel-gs)
    - [Setup](#setup)
    - [Training](#training)
    - [Render Pretrained-Model](#rendering)
    - [Calculate Metrics](#evaluating-metrics)
    - [Migrating from original 3DGS codebase](#migrating-from-original-3dgs-codebase)
- [Benefits and Examples](#benefits-and-examples)
- [Paper](#paper-and-citation)
- [License](#license)
- [Reference](#reference)
------

# Why use Grendel-GS

Here is a diagram showing why you may need distributed gaussian splatting training like our Grendel-GS' techniques:


![whydistributed](https://github.com/nyu-systems/Grendel-GS/assets/45677459/4237777e-662c-4611-9196-f60fcae927d6)



# How to use Grendel-GS

This repo and its dependency, our customized distributed version of rendering cuda code([diff-gaussian-rasterization](https://github.com/nyu-systems/diff-gaussian-rasterization)), are both forks from the [original 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting). Therefore, the usage is generally very similar to the original 3DGS. 

The two main differences are:

1. We support training on multiple GPUs, using the `torchrun` command-line utility provided by PyTorch to launch jobs.
2. We support batch sizes greater than 1, with the `--bsz` argument flag used to specify the batch size.


<!-- 
Our repository contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions to produce trained models. 
-->

## Setup

### Cloning the Repository

The repository contains submodules, thus please check it out with 
```shell
git clone git@github.com:nyu-systems/Grendel-GS.git --recursive
```

### Pytorch Environment

Ensure you have Conda, GPU with compatible driver and cuda environment installed on your machine, as prerequisites. Then please install `PyTorch`, `Torchvision`, `Plyfile`, `tqdm` which are essential packages. Make sure PyTorch version >= 1.10 to have torchrun for distributed training. Finally, compile and install two dependent cuda repo `diff-gaussian-rasterization` and `simple-knn` containing our customized cuda kernels for rendering and etc.

We provide a yml file for easy environment setup. However, you should choose the versions to match your local running environment. 
```
conda env create --file environment.yml
conda activate gaussian_splatting
```

NOTES: We kept additional dependencies minimal compared to the original 3DGS. For environment setup issues, maybe you could refer to the [original 3DGS repo issue section](https://github.com/graphdeco-inria/gaussian-splatting/issues) first.

### Reproducing this repository (including the local recursive scripts)

Reproduction on another machine is feasible, but it is not fully containerized: Grendel compiles CUDA extensions locally, so the NVIDIA driver, PyTorch CUDA build, CUDA toolkit (`nvcc`), compiler, and GPU architecture must be compatible. Record the repository commit and submodule revisions together with experiment results (`git rev-parse HEAD` and `git submodule status`).

#### 1. Host requirements and checkout

Use a Linux machine with an NVIDIA GPU, a working NVIDIA driver/NVIDIA Container Toolkit if using Docker, and a CUDA-capable PyTorch installation. For a source checkout, include all three CUDA submodules:

```bash
git clone --recursive <Grendel-GS-repository-url>
cd Grendel-GS
git submodule update --init --recursive
nvidia-smi
```

The COLMAP dataset must be prepared before training. A typical scene directory contains `images/` and `sparse/` (COLMAP format); the recursive scripts discover scenes under a dataset root.

#### 2. Python and CUDA extension build

The checked-in `environment.yml` specifies Python 3.8, PyTorch 2.2.0, torchvision 0.17.0, and PyTorch CUDA 12.1. Create and activate it with:

```bash
conda env create -f environment.yml
conda activate Grendel
```

The local `build.sh` currently expects CUDA Toolkit 12.4 at `/usr/local/cuda-12.4` and GCC/G++ 11. It builds `simple-knn`, the customized `diff-gaussian-rasterization`, and `gsplat`:

```bash
which nvcc
nvcc --version
gcc-11 --version
python -c 'import torch; print("torch", torch.__version__, "torch CUDA", torch.version.cuda, "available", torch.cuda.is_available())'
./build.sh
```

Before building on a different machine, check that `CUDA_HOME`, `nvcc`, the PyTorch CUDA version, and the compiler are mutually compatible. In particular, the supplied Conda environment uses PyTorch CUDA 12.1 while `build.sh` selects Toolkit 12.4; CUDA extension builds can warn or fail when these differ. The supplied `Dockerfile.runner` is **not** a complete Grendel build/training image: it is a Vulkan application runtime based on a CUDA *runtime* image, and does not install PyTorch, `nvcc`, or the Grendel Python CUDA extensions.

This repository includes `Dockerfile.grendel-runner`, a separate CUDA development image that installs the Conda environment and builds the extensions. Build it from the repository root after initializing the submodules:

```bash
git submodule update --init --recursive
docker build -f Dockerfile.grendel-runner -t grendel-runner:local .
docker run --rm --gpus all --ipc=host -it \
  -v /data:/data -v /output:/output \
  grendel-runner:local
```

The image embeds this checkout; mount datasets and output directories at paths visible inside the container. `TORCH_CUDA_ARCH_LIST` is set for common NVIDIA architectures (Volta through Hopper); override it at build time if the target GPU has a different compute capability, for example `--build-arg TORCH_CUDA_ARCH_LIST="8.6"`.

CUDA compatibility notes: the container does **not** need the same CUDA Toolkit installed on the host. The container supplies its own user-space toolkit/libraries, while `--gpus all` exposes the host GPU and injects the host NVIDIA driver interface. The host must have a working NVIDIA driver and NVIDIA Container Toolkit, and that driver must be new enough for the container's CUDA stack. CUDA 12.x minor-version compatibility lists Linux driver 525.60.13 as a baseline, but CUDA 12.4's corresponding driver is 550.54.14; using the latter or newer is the safer choice, especially for features outside minor-version compatibility. See NVIDIA's [CUDA compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) and [CUDA 12.4 release notes](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-toolkit-release-notes/index.html). The Conda environment currently provides PyTorch CUDA 12.1 while the image's compiler toolkit is CUDA 12.4, matching the existing build setup; extension compilation may emit a minor-version warning. If compilation or runtime fails, align PyTorch's CUDA build and the container toolkit, rebuild all extensions inside the image, and avoid copying compiled `.so` files between different PyTorch/CUDA environments. This image has not been validated on every GPU/driver combination.

Check the Python entry points after the extension build:

```bash
python train.py --help
python recursive_render.py --help
```

#### 3. Train, merge, and benchmark

For a two-GPU single-scene training run (replace the path with a prepared COLMAP scene):

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=2 train.py \
  -s /data/tanksandtemples/truck \
  -m /output/truck \
  --bsz 1 --eval --iterations 30000 --resolution -1
```

To train each recognized scene directly under a dataset root with the repository's recursive settings:

```bash
./train_recursive.sh --recursive /data/tanksandtemples
```

Merge distributed point-cloud shards before rendering. The merge script currently looks for `iteration_30000` by default; make that match the checkpoint produced by your training run:

```bash
./recursive_merge.sh /data/tanksandtemples/truck /data/tanksandtemples/truck/trained
```

Then benchmark the merged model without saving rendered PNGs; FPS/latency results are written to CSV under the model's point-cloud iteration output:

```bash
torchrun --standalone --nnodes=1 --nproc-per-node=1 recursive_render.py \
  /data/tanksandtemples/truck \
  /data/tanksandtemples/truck/trained \
  --iteration 30000 --resolutions 1.0 0.5 3840x2160 --no-save-renders
```

If using `train_recursive.sh`, check its configured saved iteration and align `recursive_merge.sh` and `recursive_render.py --iteration` with the actual `point_cloud/iteration_<N>` directory. For repeatable FPS comparisons, keep the GPU model/count, driver, power mode, resolution list, camera trace, warm-up/measurement settings, and repository/submodule revisions fixed; ensure no other GPU workload is running. Training memory use and the final Gaussian count depend strongly on the scene and GPU capacity, so a run that fits one machine may need lower input resolution or adjusted training settings on another.

### Dataset

We use colmap format to load dataset. Therefore, please download and unzip colmap datasets before trainning, for example [Mip360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) and 4 scenes from [Tanks&Temple and DeepBlending](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). 



<!-- Its core idea is to leverage more GPU's capability during training and to increase the batch size to utilize these GPU better. Therefore, we could accommodate much more gaussian primitives in large-scale and high resolution scenes, and speed up at the same time. It contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models.  -->





<!--     
_Authors: [**Hexu Zhao¹**](https://tarzanzhao.github.io), [**Haoyang Weng¹\***](https://egalahad.github.io), [**Daohan Lu¹\***](https://daohanlu.github.io), [**Ang Li²**](https://www.angliphd.com), [**Jinyang Li¹**](https://www.news.cs.nyu.edu/~jinyang/), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/), [**Saining Xie¹**](https://www.sainingxie.com)_  (\* *Indicates equal contribution*)

_Affiliations: [**¹New York University**](https://cs.nyu.edu/home/index.html), [**²Pacific Northwest National Laboratory**](https://www.pnnl.gov)_
- **[Arxiv Paper](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fp6/03-05-2024/README.md)**
- **[Pre-trained Models (14 GB)](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fp6/03-05-2024/README.md)**
- **[Evaluation Images (7 GB)](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/evaluation/images.zip)**


(TODO: check all the links here before releasing)

Abstract: *3D Gaussian Splatting (3DGS) is gaining popularity for 3D reconstruction because of its superior visual quality and rendering speed. However, training is currently done on a single GPU, and thus cannot handle high-resolution and large-scale 3D reconstruction tasks due to the GPU's memory capacity limit. We build a distributed system, called Grendel, to partition 3DGS' parameters and parallelize its computation across multiple GPUs. As each Gaussian affects a small and changing subset of rendered pixels, Grendel relies on sparse all-to-all communication to transfer each required Gaussian to a pixel partition and performs dynamic load balancing. Unlike existing 3DGS systems that train using one camera view image at a time, Grendel supports batched training using multiple views. We explore different learning rate scaling strategies and identify the simple sqrt(batch size) scaling rule to be highly effective. Evaluation using large-scale high-resolution scenes show that Grendel can improve rendering quality by scaling up 3DGS quantity using multiple GPUs. On the "Rubble" dataset, we achieve a test PSNR of 27.28 by distributing 40.4 million Gaussians across 16 GPUs. By comparison, one achieves a PSNR of 26.28 when using 11.2 million Gaussians in order to fit in a single GPU's memory.*


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
(TODO:change it to our bib)
</section> 
-->

<!--     
---

# Repository Overview
Our repository is a research-focused framework for distributed Gaussian Splatting training, as detailed in the paper. Its core idea is to leverage more GPU by distributed computation during training and to increase the batch size to utilize these GPU better. Therefore, we could accommodate much more gaussian primitives in large-scale and high resolution scenes, and speed up at the same time. It contains a distributed PyTorch-based optimizer to produce a 3D Gaussian model from SfM inputs. The optimizer uses PyTorch and CUDA extensions in a Python environment to produce trained models. 

Advantages of using our distributed implementation for gaussians splatting:
1. Train at higher resolution.
2. Train larger scenes.
3. Train the same scene dozens of times faster.
4. Increased PSNR with the same training time.
-->



## Training

For single-GPU non-distributed training with batch size of 1:
```shell
python train.py -s <path to COLMAP dataset> --eval
```

For 4 GPU distributed training and batch size of 4:
```shell
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --bsz 4 -s <path to COLMAP dataset> --eval
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for train.py</span></summary>

  #### --source_path / -s
  Path to the source directory containing a COLMAP data set.
  #### --model_path / -m 
  Path where the trained model and loggings should be stored (```/tmp/gaussian_splatting``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --bsz
  The batch size(the number of camera views) in single step training. ```1``` by default.
  #### --backend
  The CUDA backend to use in training. Valid options include ```diff``` ([diff-gaussian-rasterization](https://github.com/nyu-systems/diff-gaussian-rasterization)) and ```gsplat```. ```diff``` by default.
  #### --lr_scale_mode
  The mode of scaling learning rate given larger batch size. ```sqrt``` by default.
  #### --preload_dataset_to_gpu
  Save all groundtruth images from the dataset in GPU, rather than load each image on-the-fly at each training step. 
  If dataset is large, preload_dataset_to_gpu will lead to OOM; when the dataset is small, preload_dataset_to_gpu could 
  speed up the training a little bit by avoiding some cpu-gpu communication. 
  #### --iterations
  Number of total iterations to train for, ```30_000``` by default.
  #### --test_iterations
  Space-separated iterations at which the training script computes L1 and PSNR over test set, ```7000 30000``` by default.
  #### --save_iterations
  Space-separated iterations at which the training script saves the Gaussian model, ```7000 30000 <iterations>``` by default.
  #### --checkpoint_iterations
  Space-separated iterations at which to store a checkpoint for continuing later, saved in the model directory.
  #### --start_checkpoint
  Path to a saved checkpoint to continue training from.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.
  #### --sh_degree
  Order of spherical harmonics to be used (no larger than 3). ```3``` by default.
  #### --feature_lr
  Spherical harmonics features learning rate, ```0.0025``` by default.
  #### --opacity_lr
  Opacity learning rate, ```0.05``` by default.
  #### --scaling_lr
  Scaling learning rate, ```0.005``` by default.
  #### --rotation_lr
  Rotation learning rate, ```0.001``` by default.
  #### --position_lr_max_steps
  Number of steps (from 0) where position learning rate goes from ```initial``` to ```final```. ```30_000``` by default.
  #### --position_lr_init
  Initial 3D position learning rate, ```0.00016``` by default.
  #### --position_lr_final
  Final 3D position learning rate, ```0.0000016``` by default.
  #### --position_lr_delay_mult
  Position learning rate multiplier (cf. Plenoxels), ```0.01``` by default. 
  #### --densify_from_iter
  Iteration where densification starts, ```500``` by default. 
  #### --densify_until_iter
  Iteration where densification stops, ```15_000``` by default.
  #### --densify_grad_threshold
  Limit that decides if points should be densified based on 2D position gradient, ```0.0002``` by default.
  #### --densification_interval
  How frequently to densify, ```100``` (every 100 iterations) by default.
  #### --opacity_reset_interval
  How frequently to reset opacity, ```3_000``` by default. 
  #### --lambda_dssim
  Influence of SSIM on total loss from 0 to 1, ```0.2``` by default. 
  #### --percent_dense
  Percentage of scene extent (0--1) a point must exceed to be forcibly densified, ```0.01``` by default.



</details>
<br>

## Rendering

```shell
python render.py -s <path to COLMAP dataset> --model_path <path to folder of saving model> 
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for render.py</span></summary>

  #### --model_path / -m 
  Path to the trained model directory you want to create renderings for.
  #### --skip_train
  Flag to skip rendering the training set.
  #### --skip_test
  Flag to skip rendering the test set.
  #### --distributed_load
  If point cloud models are saved distributedly during training, we should set this flag to load all of them.
  #### --quiet 
  Flag to omit any text written to standard out pipe. 

  **The below parameters will be read automatically from the model path, based on what was used for training. However, you may override them by providing them explicitly on the command line.** 

  #### --source_path / -s
  Path to the source directory containing a COLMAP or Synthetic NeRF data set.
  #### --images / -i
  Alternative subdirectory for COLMAP images (```images``` by default).
  #### --eval
  Add this flag to use a MipNeRF360-style training/test split for evaluation.
  #### --llffhold
  The training/test split ratio in the whole dataset for evaluation. llffhold=8 means 1/8 is used as test set and others are used as train set.
  #### --white_background / -w
  Add this flag to use white background instead of black (default), e.g., for evaluation of NeRF Synthetic dataset.

</details>

For interactive rendering, please refer to [GaussFusion](https://github.com/EGalahad/GaussFusion), which also support rendering two checkpoints with interactive controls.

## Evaluating metrics

```shell
python metrics.py --model_path <path to folder of saving model> 
```

<details>
<summary><span style="font-weight: bold;">Command Line Arguments for metrics.py</span></summary>

  #### --model_paths / -m 
  Space-separated list of model paths for which metrics should be computed.
</details>
<br>

## Migrating from original 3DGS codebase

If you are currently using the original 3DGS codebase for training in your application, you can effortlessly switch to our codebase because we haven't made any algorithmic changes. This will allow you to train faster and successfully train larger, higher-precision scenes without running out of memory (OOM) within a reasonable time frame. 

It is worth noting that we only support the training functionality; this repository does not include the interactive viewer, network viewer, or colmap features from the original 3DGS. We are actively developing to support more features. Please let us know your needs or directly contribute to our project. Thank you!

---



# Benefits and Examples

## Significantly Faster Training Without Compromising Reconstruction Quality On Mip360 Dataset

### Training Time

| 30k Train Time(min)   |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:----------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1  |   24.03 |     30.18 |     25.58 |  22.45 |     21.6  |    30.15 |    19.18 |
| 4 GPU + Batch Size=1  |    9.07 |     11.67 |      9.53 |   8.93 |      8.82 |    10.85 |     8.03 |
| 4 GPU + Batch Size=4  |    5.22 |      6.47 |      6.98 |   6.18 |      5.98 |     6.48 |     5.28 |

### Test PSNR

| 30k Test PSNR        |   stump |   bicycle |   kitchen |   room |   counter |   garden |   bonsai |
|:---------------------|--------:|----------:|----------:|-------:|----------:|---------:|---------:|
| 1 GPU + Batch Size=1 |   26.61 |     25.21 |     31.4  |  31.4  |     28.93 |    27.27 |    32.01 |
| 4 GPU + Batch Size=1 |   26.65 |     25.19 |     31.41 |  31.38 |     28.98 |    27.28 |    31.92 |
| 4 GPU + Batch Size=4 |   26.59 |     25.17 |     31.37 |  31.32 |     28.98 |    27.2  |    31.94 |
---

### Reproduction Instructions

1. Download and unzip the [Mip360 dataset](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip).
2. Activate the appropriate conda/python environment.
3. To execute all experiments and generate this table, run the following command:
   ```bash
   bash examples/mip360/eval_all_mip360.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Significantly Speed up and Reduce per-GPU memory usage on Mip360 at *4K Resolution*

| Configuration                  | 50k Training Time   |   Memory Per GPU |   PSNR |
|:-------------------------------|:--------------------|-----------------:|-------:|
| bicycle + 1 GPU + Batch Size=1 | 2h 38min            |            37.18 |  23.78 |
| bicycle + 4 GPU + Batch Size=1 | 0h 50min            |            10.39 |  23.79 |
| garden + 1 GPU + Batch Size=1  | 2h 49min            |            29.87 |  26.06 |
| garden + 4 GPU + Batch Size=1  | 0h 50min            |             7.88 |  26.06 |

Unlike the typical approach of downsampling the Mip360 dataset by a factor of four before training, our system can train directly at full resolution. The bicycle and garden images have resolutions of 4946x3286 and 5187x3361, respectively. Our distributed system demonstrates that we can significantly accelerate and reduce memory usage per GPU by several folds without sacrificing quality.

### Reproduction Instructions

Set up the dataset and Python environment as outlined previously, then execute the following:
```bash
   bash examples/mip360_4k/eval_mip360_4k.sh <path_to_save_experiment_results> <path_to_mip360_dataset>
   ```

## Train in 45 Seconds on Tanks&Temple at *1K Resolution*

| Configuration                | 7k Training Time   |   7k test PSNR | 30k Training Time   |   30k test PSNR |
|:-----------------------------|:-------------------|---------------:|:--------------------|----------------:|
| train + 4 GPU + Batch Size=8 | 44s                |          19.37 | 3min 30s            |           21.87 |
| truck + 4 GPU + Batch Size=8 | 45s                |          23.79 | 3min 39s            |           25.35 |

Tanks&Temples dataset includes train and truck scenes with resolutions of 980x545 and 979x546, respectively. Utilizing 4 GPUs, we've managed to train on these small scenes to a reasonable quality in just 45 seconds(7k iterations). In the original Gaussian splatting papers, achieving a test PSNR of 18.892 and 23.506 at 7K resolution was considered good on train and truck, respectively. Our results are comparable to these benchmarks.

### Reproduction Instructions

Set up the [Tanks&Temple and DeepBlending Dataset](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) and Python environment as outlined previously, then execute the following:
```bash
   bash examples/train_truck_1k/eval_train_truck_1k.sh <path_to_save_experiment_results> <path_to_tandb_dataset>
   ```

(TODO: check these scripts have no side-effects)

## Experimental Setup for all experiments statistics above

- **Hardware**: 4x 40GB NVIDIA A100 GPUs
- **Interconnect**: Fully-connected Bidirectional 25GB/s NVLINK

---



# New features [Please check regularly!]

- We will release our optimized cuda kernels within gaussian splatting soon for further speed up. 
- We will support gsplat later as another choice of our cuda kernel backend. 

# Paper and Citation

Our system design, analysis of large-batch training dynamics, and insights from scaling up are all documented in the paper below: 

> [**On Scaling Up 3D Gaussian Splatting Training**](https://arxiv.org/abs/2406.18533)<br>
> [**Hexu Zhao¹**](https://tarzanzhao.github.io), [**Haoyang Weng¹\***](https://egalahad.github.io), [**Daohan Lu¹\***](https://daohanlu.github.io), [**Ang Li²**](https://www.angliphd.com), [**Jinyang Li¹**](https://www.news.cs.nyu.edu/~jinyang/), [**Aurojit Panda¹**](https://cs.nyu.edu/~apanda/), [**Saining Xie¹**](https://www.sainingxie.com)  (\* *co-second authors*)
> <br>¹New York University, ²Pacific Northwest National Laboratory <br>

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@misc{zhao2024scaling3dgaussiansplatting,
      title={On Scaling Up 3D Gaussian Splatting Training}, 
      author={Hexu Zhao and Haoyang Weng and Daohan Lu and Ang Li and Jinyang Li and Aurojit Panda and Saining Xie},
      year={2024},
      eprint={2406.18533},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2406.18533}, 
}</code></pre>
  </div>
</section> 

# Code Specification
Please use "black" with default settings to format the code if you want to contribute.
## Setup
```shell
conda install black==24.4.2
```

# License

Distributed under the Apache License Version 2.0 License. See `LICENSE.txt` for more information.

# Reference

1. Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. ACM Transactions on Graphics, July 2023. URL: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/.
