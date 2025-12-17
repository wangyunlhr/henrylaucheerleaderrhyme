import math
import sys

sys.path.append('./')

import os, argparse, glob, datetime, yaml
import torch
from torch.utils.data import DataLoader
import time
import numpy as np
from tqdm import tqdm
import joblib

from omegaconf import OmegaConf
from PIL import Image

from lidm.utils.misc_utils import instantiate_from_config, set_seed
from lidm.utils.lidar_utils import range2pcd, range2point_kitticropped
from lidm.eval.eval_utils import evaluate
from lidm.eval.eval_utils import evaluate_range_image

# remove annoying user warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
#! 
from lidm.data.kitti import HDF5Dataset, HDF5Dataset_kitti
import torch.nn.functional as F
import pytorch_lightning as pl
import open3d as o3d
import re



DATASET2METRICS = {'kitti': ['frid', 'fsvd', 'fpvd', 'jsd', 'mmd'], 'nuscenes': ['fsvd', 'fpvd']}
DATASET2TYPE = {'kitti': '64', 'nuscenes': '32'}

custom_to_range = lambda x: (x * 255.).clamp(0, 255).floor() / 255.


def custom_to_pcd(x, config, rgb=None):
    x = x.squeeze().detach().cpu().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    if rgb is not None:
        rgb = rgb.squeeze().detach().cpu().numpy()
        rgb = (np.clip(rgb, -1., 1.) + 1.) / 2.
        rgb = rgb.transpose(1, 2, 0)
    xyz, rgb, _ = range2pcd(x, color=rgb, **config['data']['params']['dataset'])

    return xyz, rgb


def custom_to_pil(x):
    x = x.detach().cpu().squeeze().numpy()
    x = (np.clip(x, -1., 1.) + 1.) / 2.
    x = (255 * x).astype(np.uint8)

    if x.ndim == 3:
        x = x.transpose(1, 2, 0)
    x = Image.fromarray(x)

    return x


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs

def range2point(
        range_map,                  # (H, W_crop)  当前帧裁剪后的 range image（米）
        sensor_center,              # (3,)         传感器中心（世界系）
        sensor2world,               # (4, 4)       从传感器到世界的位姿矩阵
        ir,                         # 俯仰角信息：形如 [-ir_min, ir_max] 或 (2,)；或逐行数组 (H,)
        angle_offset,               # 标量，方位角零点偏移（弧度）
        pixel_offset,
        mask
    ):
    """
    return a tensor of points (H, W, 3) in the world coordinate in HOST
    """

    # data preprocess
    if not torch.is_tensor(range_map):
        range_map = torch.tensor(range_map, dtype=torch.float32, device="cuda")
    if range_map.dim() != 2:
        if range_map.dim() == 3:
            if range_map.shape[0] == 1:
                range_map = range_map[0]
            elif range_map.shape[2] == 1:
                range_map = range_map[..., 0]
            else:
                raise ValueError("range_map is not (H, W, 1) or (1, H, W)")
        else:
            raise ValueError("range_map shape unindentified")
    if not torch.is_tensor(sensor_center):
        sensor_center = torch.tensor(
            sensor_center, dtype=torch.float32, device="cuda"
        )
    if not torch.is_tensor(sensor2world):
        sensor2world = torch.tensor(
            sensor2world, dtype=torch.float32, device="cuda"
        )

    H, W = range_map.shape
    rays_o = sensor_center.cuda()[None, None, ...].expand(H, W, 3)

    y = torch.ones(H, device="cuda", dtype=torch.float32)
    x = (
        torch.arange(W, 0, -1, device="cuda", dtype=torch.float32)
        - pixel_offset
    ) / float(W)
    grid_y, grid_x = torch.meshgrid(y, x)

    azimuth = grid_x * 2 * torch.pi - torch.pi - angle_offset
    ir = ir.tolist()
    # if type(ir) != list and type(ir) != tuple:
    #     ir = [-ir, ir]
    if len(ir) == 2:
        grid_y = (
            grid_y
            * (
                torch.arange(
                    H, 0, -1, device="cuda", dtype=torch.float32
                ).unsqueeze(-1)
                - pixel_offset
            )
            / float(H)
        )
        inclination = grid_y * (ir[1] - ir[0]) + ir[0]
    else:
        inclination = grid_y * torch.tensor(
            (ir), device="cuda", dtype=torch.float32
        ).flip(0).unsqueeze(-1)
    rays_x = torch.cos(inclination) * torch.cos(azimuth)
    rays_y = torch.cos(inclination) * torch.sin(azimuth)
    rays_z = torch.sin(inclination)

    rays_d = torch.stack([rays_x, rays_y, rays_z], dim=-1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    points = rays_d * range_map[..., None].cuda()
    # points = points @ sensor2world[:3, :3].T.cuda() + sensor2world[:3, 3].cuda()
    pts = points
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float().to(pts.device)
    if len(mask.shape) == 2:
        mask =(mask == 1)
        pts = pts[mask]
    elif len(mask.shape) == 3:
        pts = pts * mask
    return pts # sensor坐标系下的点云

def depth_restore_value_tensor(depth: torch.Tensor, depth_max: float = 80.0, safe_fp32: bool = True) -> torch.Tensor:
    """
    depth: [-1,1] 的张量，形状任意；设备可为 CPU 或 CUDA。
    depth_max: 恢复到米制的最大深度。
    safe_fp32: 在 bf16/fp16 下先用 fp32 计算更稳，最后再转回原 dtype。
    """
    orig_dtype = depth.dtype
    x = depth.to(torch.float32) if (safe_fp32 and orig_dtype in (torch.bfloat16, torch.float16)) else depth

    # [-1,1] -> [0,1]
    depth_01 = (x + 1.0) * 0.5
    # 夹紧到 [0,1]
    depth_01 = depth_01.clamp(0.0, 1.0)
    # [0, depth_max]
    out = depth_01 * float(depth_max)

    return out.to(orig_dtype) if x is not depth else out



def batch_range2point_from_batch(batch, range_maps=None, resize_flag=False, device="cuda"):
    """
    输入：一个 dataloader 给出的 batch（字典形式）
    输出：point_list，长度为 B 的 list，每个元素是 (N_i, 3) 的点云（在 sensor 坐标系）
    """
    # 这里我用 image_ori 当 range_map，如果你想用 segmentation，也可以把下面的 key 换掉

    range_maps = depth_restore_value_tensor(range_maps.to(device), depth_max=80.0) # (B, 1, H, W) 或 (B, H, W)
    gt_range_maps = range_maps.detach().cpu().numpy()
    sensor_centers = batch["sensor_center"].to(device) # (B, 3)
    sensor2worlds = batch["sensor2world"].to(device)   # (B, 4, 4)
    range_mask = batch["mask_ori"].to(device) 
    # irs = batch["ir"]                                  # 可能是 Tensor or np
    # angle_offsets = batch["angle_offset"]              # (B,)
    # pixel_offsets = batch["pixel_offset"]              # (B,)

    B = range_maps.shape[0]
    point_list = []

    for i in range(B):
        # ---- 1) 取出第 i 个样本 ----
        range_map_i = range_maps[i]
        # 保证变成 (H, W) 形状，因为你的 range2point 是按单帧写的
        if range_map_i.dim() == 3:  # (1,H,W) -> (H,W)
            range_map_i = range_map_i[0]
            range_mask_i = range_mask[i][0]

        sensor_center_i = sensor_centers[i]   # (3,)
        sensor2world_i = sensor2worlds[i]     # (4,4)
        # ir_i = irs[i]                         # (2,) 或 (H,) 等
        # angle_offset_i = angle_offsets[i]
        # pixel_offset_i = pixel_offsets[i]
        ir =  (math.radians(-24.9), math.radians(2.0))
        # 如果 angle_offset/pixel_offset 是 0-D tensor，就取 item()
        # if torch.is_tensor(angle_offset_i):
        #     angle_offset_i = angle_offset_i.item()
        # if torch.is_tensor(pixel_offset_i):
        #     pixel_offset_i = pixel_offset_i.item()

        # ---- 2) 调用你原来的 range2point ----
        points_hw3 = range2point_kitticropped(
                                            range_map_i,                 # cropped: (64, 1024) or (1,64,1024) or (64,1024,1)
                                            ir,
                                            sensor_center_i,
                                            sensor2world_i,
                                            )  # -> (H, W, 3) in sensor frame

        # 按需保存为 cpu / numpy
        point_list.append(points_hw3.detach().cpu().numpy())

    return point_list, gt_range_maps


def single_range2point_from_sample(sample, index, range_map=None, device="cuda"):
    """
    输入：
        sample: dataloader 里取出的单个样本（原来 batch 里的一个元素），字典形式：
            {
                "image_ori": (1, H, W) 或 (H, W)，[-1,1] 归一化的 depth
                "mask_ori":  (1, H, W) 或 (H, W)，0/1 mask（可选）
                "sensor_center": (3,)
                "sensor2world":  (4, 4)
                "ir":            标定俯仰信息
                "angle_offset":  标量
                "pixel_offset":  标量
                ...
            }
        range_map: 如果不为 None，则使用这个作为 depth；否则用 sample["image_ori"]
        device: "cuda" / "cpu"

    返回：
        points: (H, W, 3) 的点云（或者你的 range2point 里再根据 mask 过滤）
    """

    # 1) 选用哪张 range map
    if range_map is None:
        range_map = sample["image_ori"][index]  # Tensor, (1,H,W) or (H,W)
    range_map = range_map.to(device)

    # 保证是 (H, W)
    if range_map.dim() == 3 and range_map.shape[0] == 1:
        range_map = range_map[0]

    # 2) 还原 depth（如果 image_ori 是 [-1,1] 范围）
    range_map = depth_restore_value_tensor(range_map, depth_max=80.0)
    # range_map = F.interpolate(range_map.unsqueeze(0).unsqueeze(0), size=(64, 2650), mode='bilinear', align_corners=False).squeeze(0).squeeze(0)
    # 3) 取 mask（可选）
    range_mask = sample.get("mask_ori", None)[index]
    if range_mask is not None:
        range_mask = range_mask.to(device)
        if range_mask.dim() == 3 and range_mask.shape[0] == 1:
            range_mask = range_mask[0]

    # 4) 取位姿和标定信息
    sensor_center = sample["sensor_center"][index].to(device)   # (3,)
    sensor2world = sample["sensor2world"][index].to(device)     # (4,4)
    # ir = sample["ir"][index]                                    # Tensor / np 等
    # angle_offset = sample["angle_offset"][index]
    # pixel_offset = sample["pixel_offset"][index]

    # 把 0-D tensor 变成标量
    # if torch.is_tensor(angle_offset):
    #     angle_offset = angle_offset.item()
    # if torch.is_tensor(pixel_offset):
    #     pixel_offset = pixel_offset.item()

    # 5) 调用你写的 range2point（已加上 mask 参数）
    ir =  (math.radians(-24.9), math.radians(2.0))
    points_hw3 = range2point_kitticropped(
                                            range_map,                 # cropped: (64, 1024) or (1,64,1024) or (64,1024,1)
                                            ir,
                                            sensor_center,
                                            sensor2world,
                                        )

    # 如需返回 numpy：
    return points_hw3.detach().cpu().numpy(), range_map.detach().cpu().numpy(), range_mask.detach().cpu().numpy()



class H5DataModule(pl.LightningDataModule):
    def __init__(self, pkl_directory, data_path, tokenizer,
                batch_size, num_workers):
        super().__init__()
        self.pkl_directory = pkl_directory
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        # 跟你原来的写法一致
        if 'KITTI' in self.data_path:
            self.train_dataset = HDF5Dataset_kitti(
                pkl_directory=self.pkl_directory,
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                eval=False,
            )
            self.val_dataset = HDF5Dataset_kitti(
                pkl_directory=self.pkl_directory,
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                eval=True
            )
        else:
            self.train_dataset = HDF5Dataset(
                pkl_directory=self.pkl_directory,
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                eval=False,
            )
            self.val_dataset = HDF5Dataset(
                pkl_directory=self.pkl_directory,
                data_path=self.data_path,
                tokenizer=self.tokenizer,
                eval=True
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            collate_fn=test_collate_fn,
        )

def infer_next_id(imglogdir, key="samples"):
    pat = os.path.join(imglogdir, f"{key}_*.png")
    ids = []
    for p in glob.glob(pat):
        m = re.search(rf"{key}_(\d+)\.png$", os.path.basename(p))
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 0


def run(model, dataloader, imglogdir, pcdlogdir, nplog=None, config=None, verbose=False, log_config={}):
    tstart = time.time()
    n_saved = infer_next_id(imglogdir, key="samples")

    all_samples, all_gt = [], []
    gt_range_maps_list = []
    sample_range_images_list = []
    sample_gt_masks_list = []
    print(f"Running conditional sampling")
    for batch in tqdm(dataloader, desc="Sampling Batches (conditional)"):
        gt_point_list, gt_range_maps = batch_range2point_from_batch(batch, range_maps = batch["image_ori"], resize_flag=False) #! range image投影后的点云
        all_gt.extend(gt_point_list) #! range image投影后的点云
        N = len(batch['image'])
        logs = model.log_images(batch, N=N, split='val', **log_config)
        # n_saved = save_logs(logs, imglogdir, pcdlogdir, N, n_saved=n_saved, config=config)
        n_saved, sample_pcd, sample_range_image, sample_gt_mask = save_logs_batch(batch, logs, imglogdir, pcdlogdir, N, n_saved=n_saved, config=config) #! 投影成点云，需要batch参数
        # all_samples.extend([custom_to_pcd(img, config)[0].astype(np.float32) for img in logs["samples"]])
        all_samples.extend(sample_pcd)
        #! for evaluate range_image
        gt_range_maps_list.extend([i for i in gt_range_maps])
        sample_range_images_list.extend(sample_range_image)
        sample_gt_masks_list.extend(sample_gt_mask)
        # break #! for debug only
    # joblib.dump(all_samples, os.path.join(nplog, f"samples.pcd"))
    #! save_gt_point

    for idx, (gt_point, sample_gt_mask) in enumerate(zip(all_gt, sample_gt_masks_list)):
        # gt_point / mask 统一转到 numpy(cpu)
        if torch.is_tensor(gt_point):
            gt_point = gt_point.detach().cpu().numpy()
        if torch.is_tensor(sample_gt_mask):
            sample_gt_mask = sample_gt_mask.detach().cpu().numpy()

        # 可选：如果 gt_point 是 (H,W,3)，先拉平成 (H*W,3)，mask 同理
        if gt_point.ndim == 3:
            gt_point = gt_point.reshape(-1, gt_point.shape[-1])
        if sample_gt_mask.ndim >= 2:
            sample_gt_mask = sample_gt_mask.reshape(-1)

        # mask 过滤
        valid = (sample_gt_mask != 0)
        gt_point_sensor = gt_point[valid]

        # 写 ply（Open3D 更稳：float64）
        gt_point_o3d = o3d.geometry.PointCloud()
        gt_point_o3d.points = o3d.utility.Vector3dVector(gt_point_sensor.astype(np.float64))

        gt_point_path = os.path.join(pcdlogdir, f"gt_{idx:06}.ply")
        o3d.io.write_point_cloud(gt_point_path, gt_point_o3d)



    print(f"Sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")



    return all_samples, all_gt, gt_range_maps_list, sample_range_images_list, sample_gt_masks_list

def save_logs_batch(batch, logs, imglogdir, pcdlogdir, num, n_saved=0, key_list=None, config=None):
    key_list = logs.keys() if key_list is None else key_list
    sample_pcd = []
    sample_range_image = []
    sample_gt_mask = []


    # 先保存一次 grid（如果存在）
    for k in ['denoise_row_x_inter', 'denoise_row_pred_x0']:
        if k in logs and k in key_list:
            x = logs[k]
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                img = custom_to_pil(x)
                imgpath = os.path.join(imglogdir, f"{k}_{n_saved:06}.png")  # 或者用 global_step
                img.save(imgpath)


    for i in range(num): #!逐个case进行处理
        for k in key_list:
            if k in ['denoise_row_x_inter', 'denoise_row_pred_x0']:
                continue

            x = logs[k][i]
            # save as image
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                img = custom_to_pil(x)
                imgpath = os.path.join(imglogdir, f"{k}_{n_saved:06}.png")
                img.save(imgpath)
            # save as point cloud
            if k in ['samples', 'inputs']:
                if config.model.params.cond_stage_key == 'segmentation':
                    xyz, range_map, range_mask = single_range2point_from_sample(batch, index=i, range_map=x, device="cuda") #! resize恢复到原分辨率
                    rgb = np.zeros_like(xyz)
                    if k == 'samples':
                        sample_pcd.append(xyz)
                        sample_range_image.append(range_map)
                        sample_gt_mask.append(range_mask)
                    # xyz, rgb = custom_to_pcd(x, config, logs['original_conditioning'][i]) 
                else:
                    xyz, rgb = custom_to_pcd(x, config)
                # pcdpath = os.path.join(pcdlogdir, f"{k}_{n_saved:06}.txt")
                # np.savetxt(pcdpath, np.hstack([xyz, rgb]), fmt='%.3f')
                point_sensor = xyz[range_mask != 0]
                point = o3d.geometry.PointCloud()
                point.points = o3d.utility.Vector3dVector(point_sensor)
                pcd_path = os.path.join(pcdlogdir, f"{k}_{n_saved:06}.ply")
                o3d.io.write_point_cloud(pcd_path, point)

        n_saved += 1
    return n_saved, sample_pcd, sample_range_image, sample_gt_mask


def save_logs(logs, imglogdir, pcdlogdir, num, n_saved=0, key_list=None, config=None):
    key_list = logs.keys() if key_list is None else key_list
    for i in range(num):
        for k in key_list:
            if k in ['reconstruction']:
                continue
            x = logs[k][i]
            # save as image
            if x.ndim == 3 and x.shape[0] in [1, 3]:
                img = custom_to_pil(x)
                imgpath = os.path.join(imglogdir, f"{k}_{n_saved:06}.png")
                img.save(imgpath)
            # save as point cloud
            if k in ['samples', 'inputs']:
                if config.model.params.cond_stage_key == 'segmentation':
                    xyz, rgb = custom_to_pcd(x, config, logs['original_conditioning'][i])
                else:
                    xyz, rgb = custom_to_pcd(x, config)
                pcdpath = os.path.join(pcdlogdir, f"{k}_{n_saved:06}.txt")
                np.savetxt(pcdpath, np.hstack([xyz, rgb]), fmt='%.3f')
        n_saved += 1
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
        default="none"
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "--vanilla",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=10
    )
    parser.add_argument(
        "-f",
        "--file",
        help="the file path of samples",
        default=None
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        help="the numpy file path",
        default=1000
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        help="dataset name [nuscenes, kitti]",
        required=True
    )
    parser.add_argument(
        "--baseline",
        default=False,
        action='store_true',
        help="baseline provided by other sources (default option is not baseline)?",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=False,
        action='store_true',
        help="print status?",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action='store_true',
        help="evaluation results?",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd, strict=True) #! 完全加载模型参数
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model, pl_sd["state_dict"])
    return model, global_step


def visualize(samples, logdir):
    pcdlogdir = os.path.join(logdir, "pcd")
    os.makedirs(pcdlogdir, exist_ok=True)
    for i, pcd in enumerate(samples):
        # save as point cloud
        pcdpath = os.path.join(pcdlogdir, f"{i:06}.txt")
        np.savetxt(pcdpath, pcd, fmt='%.3f')


# def test_collate_fn(data):
#     output = {}
#     keys = data[0].keys()
#     for k in keys:
#         v = [d[k] for d in data]
#         if k not in ['reproj', 'raw']:
#             v = torch.from_numpy(np.stack(v, 0))
#         else:
#             v = [d[k] for d in data]
#         output[k] = v
#     return output

def test_collate_fn(batch):
    """
    batch: List[dict]，每个 dict 就是你 __getitem__ 返回的 out
    """
    output = {}
    keys = batch[0].keys()
    for k in keys:
        vals = [b[k] for b in batch]
        v0 = vals[0]

        # 1. 已经是 torch.Tensor 的，直接 stack
        if isinstance(v0, torch.Tensor):
            output[k] = torch.stack(vals, dim=0)

        # 2. numpy 数组，先 np.stack 再转 torch
        elif isinstance(v0, np.ndarray):
            output[k] = torch.from_numpy(np.stack(vals, axis=0))

        # 3. 标量数值（float/int/np.number），变成一维 tensor
        elif isinstance(v0, (float, int, np.number)):
            output[k] = torch.tensor(vals)
        # 4. 其他类型（字符串、字典等）就保留为 list
        else:
            # 例如 caption, scene_id, timestamp 这里就会是 List[str] / List[int]
            output[k] = vals

    return output





def traverse_collate_fn(data):
    pcd_list = [example['reproj'].astype(np.float32) for example in data]
    return pcd_list


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None
    set_seed(opt.seed)

    if not os.path.exists(opt.resume) and not os.path.exists(opt.file):
        raise FileNotFoundError
    if os.path.isfile(opt.resume):
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    elif os.path.isfile(opt.file):
        try:
            logdir = '/'.join(opt.file.split('/')[:-5])
            if len(logdir) == 0:
                logdir = '/'.join(opt.file.split('/')[:-1])
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -5  # take a guess: path/to/logdir/samples/step_num/date/numpy/*.npz
            logdir = "/".join(paths[:idx])
        ckpt = None
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    # if not opt.baseline: #! 
    #     base_configs = [f'{logdir}/fixed_based_sam_vaecondition_vaetrain.yaml']
    # else:
    #     base_configs = [f'models/baseline/{opt.dataset}/template/config.yaml']
    base_configs = ['/data0/code/LiDAR-Diffusion-main/configs/lidar_diffusion/kitti/fixed_based_sam.yaml']
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True
    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    if opt.file is None:
        model, global_step = load_model(config, ckpt)
        print(f"global step: {global_step}")
        print(75 * "=")
        print("logging to:")
        logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
        imglogdir = os.path.join(logdir, "img")
        pcdlogdir = os.path.join(logdir, "pcd")
        numpylogdir = os.path.join(logdir, "numpy")

        os.makedirs(imglogdir)
        os.makedirs(pcdlogdir)
        os.makedirs(numpylogdir)
        print(logdir)
        print(75 * "=")

        # write config out
        sampling_file = os.path.join(logdir, "sampling_config.yaml")
        sampling_conf = vars(opt)

        with open(sampling_file, 'w') as f:
            yaml.dump(sampling_conf, f, default_flow_style=False)
        print(sampling_conf)

        #!原始数据 traverse all validation data
        # data_config = config['data']['params']['validation']
        # data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
        #                               'aug_config': config['data']['params']['aug'], 'return_pcd': True,
        #                               'max_objects_per_image': 5})
        # dataset = instantiate_from_config(data_config)
        # dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False, drop_last=False,
        #                         collate_fn=test_collate_fn)


        #!!! NEW Dataloader
        pkl_directory = config.data.params.pkl_directory
        dataset_path = config.data.params.dataset_path
        num_workers = getattr(config.data.params, "num_workers", 8)

        # 假设model 里有 tokenizer（类似 net_difix.tokenizer）
        tokenizer = getattr(model, "tokenizer", None)

        data = H5DataModule(
            pkl_directory=pkl_directory,
            data_path=dataset_path,
            tokenizer=tokenizer,
            batch_size=opt.batch_size,
            num_workers=num_workers
        )
        data.setup()
        dataset = data.val_dataset
        dataloader = data.val_dataloader()


        # settings
        log_config = {'sample': True, 'ddim_steps': opt.custom_steps,
                      'quantize_denoised': False, # ddim去噪，但是中间过程经过了量化
                      'inpaint': False, # 用初值mask,验证condition的学习能力
                      'plot_progressive_rows': False, # 完整的采样步骤
                      'plot_diffusion_rows': False, #纯输入加噪声，decoder结果
                      'plot_denoise_rows': False, #展示去噪中间结果的decoder结果
                      'dset': dataset} #!设置可视化
        # test = dataset[0]
        all_samples, all_gt, gt_range_maps_list, sample_range_images_list, sample_gt_masks_list = run(model, dataloader, imglogdir, pcdlogdir, nplog=numpylogdir,
                                  config=config, verbose=opt.verbose, log_config=log_config)

        # recycle gpu memory
        del model
        torch.cuda.empty_cache()
    else:
        all_samples = joblib.load(opt.file)
        all_samples = [sample.astype(np.float32) for sample in all_samples]

        # traverse all validation data
        data_config = config['data']['params']['validation']
        data_config['params'].update({'dataset_config': config['data']['params']['dataset'],
                                      'aug_config': config['data']['params']['aug'], 'return_pcd': True})
        dataset = instantiate_from_config(data_config)
        dataloader = DataLoader(dataset, batch_size=64, num_workers=8, shuffle=False, drop_last=False,
                                collate_fn=traverse_collate_fn)
        all_gt = []
        for batch in dataloader:
            all_gt.extend(batch)

    # evaluation
    if opt.eval:
        metrics, data_type = DATASET2METRICS[opt.dataset], DATASET2TYPE[opt.dataset]
        evaluate_range_image(gt_range_maps_list, sample_range_images_list, sample_gt_masks_list)
        # evaluate(all_gt, all_samples, metrics, data_type)
        # for i in range(len(all_gt)):
        #     print(i,"="*10)
        #     if i == len(all_gt) - 1:
        #         break
        #     evaluate_range_image(gt_range_maps_list[i:i+1], sample_range_images_list[i:i+1], sample_gt_masks_list[i:i+1])
