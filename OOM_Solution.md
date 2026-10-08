(Grendel) ps@ps:~/hjf/3dgs/Grendel-GS$ ./run_matrixcity.sh
[2026-07-01 17:11:23,516] torch.distributed.run: [WARNING] 
[2026-07-01 17:11:23,516] torch.distributed.run: [WARNING] *****************************************
[2026-07-01 17:11:23,516] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-07-01 17:11:23,516] torch.distributed.run: [WARNING] *****************************************
Initializing ->  world_size: 2 rank: 1     in_node_size: 2 in_node_rank: 1
Initializing ->  world_size: 2 rank: 0     in_node_size: 2 in_node_rank: 0
try find /home/ps/hjf/3dgs/res/matrix_city/small_city/aerial_downsampled/pose/block_all/transforms_train.json [01/07 17:11:25]
Found transforms_train.json file, assuming Blender data set! [01/07 17:11:25]
Reading Training Transforms [01/07 17:11:25]
Output folder: /home/ps/hjf/3dgs/res/matrix_city/output/grendel_all [01/07 17:11:25]
try find /home/ps/hjf/3dgs/res/matrix_city/small_city/aerial_downsampled/pose/block_all/transforms_train.json [01/07 17:11:25]
Found transforms_train.json file, assuming Blender data set! [01/07 17:11:25]
Reading Training Transforms [01/07 17:11:25]
Reading Test Transforms [01/07 17:27:31]
Reading Test Transforms [01/07 17:27:31]
Generating random point cloud (300000)... [01/07 17:30:50]
[INIT] Camera centers range: x=[-9.80, 8.70] y=[-5.74, 9.01] z=[-4.89, 9.38] [01/07 17:30:50]
[INIT] Focus point: (0.21, -9.58, -11.04) [01/07 17:30:50]
[INIT] Depth range: [1.51, 31.85] [01/07 17:30:50]
[INIT] Point cloud sample range: x=[-18.36, 16.45] y=[-20.70, 5.05] z=[-24.17, 4.34] [01/07 17:30:50]
Decoding Training Cameras [01/07 17:30:52]
  6%|██████▌                                                                                                      | 340/5621 [00:41<02:38, 33.33it/s][2026-07-01 17:31:36,046] torch.distributed.elastic.multiprocessing.api: [WARNING] Sending process 2363588 closing signal SIGTERM
[2026-07-01 17:31:41,288] torch.distributed.elastic.multiprocessing.api: [ERROR] failed (exitcode: -9) local_rank: 1 (pid: 2363589) of binary: /home/ps/anaconda3/envs/Grendel/bin/python3.8
Traceback (most recent call last):
  File "/home/ps/anaconda3/envs/Grendel/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/ps/anaconda3/envs/Grendel/lib/python3.8/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/ps/anaconda3/envs/Grendel/lib/python3.8/site-packages/torch/distributed/run.py", line 812, in main
    run(args)
  File "/home/ps/anaconda3/envs/Grendel/lib/python3.8/site-packages/torch/distributed/run.py", line 803, in run
    elastic_launch(
  File "/home/ps/anaconda3/envs/Grendel/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 135, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/ps/anaconda3/envs/Grendel/lib/python3.8/site-packages/torch/distributed/launcher/api.py", line 268, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError: 
========================================================
train.py FAILED
--------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
--------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2026-07-01_17:31:36
  host      : ps
  rank      : 1 (local_rank: 1)
  exitcode  : -9 (pid: 2363589)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 2363589
======================================================== 使用DownSample的，发现依然会被Sigkill，这是主存不够吧，因为没有提到DEvice OOM。那么我觉得还是需要开发，使得代码中不直接Load全量的图片到内存中。例如我们可以用mmap？

由于Grendel内部会直接读取全量的图片到主存，所以导致OOM报错

✅ 正确解法：惰性加载 + DataLoader 缓存控制
不需要改 Grendel 底层 C++/CUDA 代码，只需修改 Python 端的 Dataset 类，将“全量预加载”改为“按需读取”。
1. 定位问题代码
在 Grendel 的 dataset 文件中（通常是 scene/dataset.py 或类似路径），找到类似这样的预加载逻辑：
python

编辑



# ❌ 典型的OOM元凶：初始化时把所有图片读进内存
self.images = []
for frame in frames:
    img = Image.open(frame["file_path"])
    self.images.append(np.array(img))  # 全部驻留RAM
2. 改为惰性加载
python

编辑



# ✅ 只存路径，训练时按需读取
class LazyImageDataset(Dataset):
    def __init__(self, frames, ...):
        self.frame_paths = [f["file_path"] for f in frames]
        # 不再预加载图片！
        
    def __getitem__(self, idx):
        # 每次采样时才解码，用完即释放
        img = Image.open(self.frame_paths[idx]).convert("RGB")
        return transforms.ToTensor()(img)
3. 关键：限制 DataLoader 预取
即使改了惰性加载，如果 DataLoader 的 num_workers 过高，每个 worker 仍会缓存大量图片。必须严格限制：
python

编辑



DataLoader(
    dataset,
    batch_size=1,           # 3DGS通常batch=1
    num_workers=2,          # ← 从默认8降到2，每个worker最多缓存2张
    prefetch_factor=1,      # ← PyTorch≥1.13可用，禁止过度预取
    persistent_workers=True # 避免反复创建worker的开销
)