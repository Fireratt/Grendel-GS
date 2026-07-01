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