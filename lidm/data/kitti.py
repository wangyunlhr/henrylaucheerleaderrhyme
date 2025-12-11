import glob
import os
import pickle
import numpy as np
import yaml
from PIL import Image
import xml.etree.ElementTree as ET

from lidm.data.base import DatasetBase
from .annotated_dataset import Annotated3DObjectsDataset
from .conditional_builder.utils import corners_3d_to_2d
from .helper_types import Annotation
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd, kitti_points_to_range_image

import torch
import h5py
import torch.nn.functional as F
# TODO add annotation categories and semantic categories
CATEGORIES = ['ignore', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist',
              'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
              'pole', 'traffic-sign']
CATE2LABEL = {k: v for v, k in enumerate(CATEGORIES)}  # 0: invalid, 1~10: categories
LABEL2RGB = np.array([(0, 0, 0), (0, 0, 142), (119, 11, 32), (0, 0, 230), (0, 0, 70), (0, 0, 90), (220, 20, 60),
                      (255, 0, 0), (0, 0, 110), (128, 64, 128), (250, 170, 160), (244, 35, 232), (230, 150, 140),
                      (70, 70, 70), (190, 153, 153), (107, 142, 35), (0, 80, 100), (230, 150, 140), (153, 153, 153),
                      (220, 220, 0)])
CAMERAS = ['CAM_FRONT']
BBOX_CATS = ['car', 'people', 'cycle']
BBOX_CAT2LABEL = {'car': 0, 'truck': 0, 'bus': 0, 'caravan': 0, 'person': 1, 'rider': 2, 'motorcycle': 2, 'bicycle': 2}

# train + test
SEM_KITTI_TRAIN_SET = ['00', '01', '02', '03', '04', '05', '06', '07', '09', '10']
KITTI_TRAIN_SET = SEM_KITTI_TRAIN_SET + ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
KITTI360_TRAIN_SET = ['00', '02', '04', '05', '06', '07', '09', '10'] + ['08']  # partial test data at '02' sequence
CAM_KITTI360_TRAIN_SET = ['00', '04', '05', '06', '07', '08', '09', '10']  # cam mismatch lidar in '02'

# validation
SEM_KITTI_VAL_SET = KITTI_VAL_SET = ['08']
CAM_KITTI360_VAL_SET = KITTI360_VAL_SET = ['03']


class KITTIBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'kitti'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        points = scan[:, 0:3]  # get xyz
        return points

    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        check_range_image = kitti_points_to_range_image(sweep, np.ones_like(sweep))
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)

        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras

        return example


class SemanticKITTIBase(KITTIBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in ['segmentation']  # for segmentation input only
        self.label2rgb = LABEL2RGB

    def prepare_data(self):
        # read data paths from KITTI
        for seq_id in eval('SEM_KITTI_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'dataset/sequences/{seq_id}/velodyne/*.bin')))
        # read label mapping
        data_config = yaml.safe_load(open('./data/config/semantic-kitti.yaml', 'r'))
        remap_dict = data_config["learning_map"]
        max_key = max(remap_dict.keys())
        self.learning_map = np.zeros((max_key + 100), dtype=np.int32)
        self.learning_map[list(remap_dict.keys())] = list(remap_dict.values())

    def load_semantic_map(self, path, pcd):
        label_path = path.replace('velodyne', 'labels').replace('.bin', '.label')
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels.reshape((-1))
        labels = labels & 0xFFFF  # semantic label in lower half
        labels = self.learning_map[labels]

        proj_range, sem_map = pcd2range(pcd, self.img_size, self.fov, self.depth_range, labels=labels)
        # sem_map = np.expand_dims(sem_map, axis=0).astype(np.int64)
        sem_map = sem_map.astype(np.int64)
        if self.filtered_map_cats is not None:
            sem_map[np.isin(sem_map, self.filtered_map_cats)] = 0  # set filtered category as noise
        onehot = np.eye(self.num_sem_cats, dtype=np.float32)[sem_map].transpose(2, 0, 1)
        return proj_range, onehot


class SemanticKITTITrain(SemanticKITTIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/SemanticKITTI', split='train', **kwargs)


class SemanticKITTIValidation(SemanticKITTIBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/SemanticKITTI', split='val', **kwargs)


class KITTI360Base(KITTIBase):
    def __init__(self, split_per_view=None, **kwargs):
        super().__init__(**kwargs)
        self.split_per_view = split_per_view
        if self.condition_key == 'camera':
            assert self.split_per_view is not None, 'For camera-to-lidar, need to specify split_per_view'

    def prepare_data(self):
        # read data paths
        self.data = []
        if self.condition_key == 'camera':
            seq_list = eval('CAM_KITTI360_%s_SET' % self.split.upper())
        else:
            seq_list = eval('KITTI360_%s_SET' % self.split.upper())
        for seq_id in seq_list:
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))

    def random_drop_camera(self, camera_list):
        if np.random.rand() < self.aug_config['camera_drop'] and self.split == 'train':
            camera_list = [np.zeros_like(c) if i != len(camera_list) // 2 else c for i, c in enumerate(camera_list)]  # keep the middle view only
        return camera_list

    def load_camera(self, path):
        camera_path = path.replace('data_3d_raw', 'data_2d_camera').replace('velodyne_points/data', 'image_00/data_rect').replace('.bin', '.png')
        camera = np.array(Image.open(camera_path)).astype(np.float32) / 255.
        camera = camera.transpose(2, 0, 1)
        if self.view_transform:
            camera = self.view_transform(camera)
        camera_list = np.split(camera, self.split_per_view, axis=2)  # split into n chunks as different views
        camera_list = self.random_drop_camera(camera_list)
        return camera_list


class KITTI360Train(KITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/KITTI-360/', split='train', **kwargs)


class KITTI360Validation(KITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/KITTI-360/', split='val', **kwargs)


class AnnotatedKITTI360Base(Annotated3DObjectsDataset, KITTI360Base):
    def __init__(self, **kwargs):
        self.id_bbox_dict = dict()
        self.id_label_dict = dict()

        Annotated3DObjectsDataset.__init__(self, **kwargs)
        KITTI360Base.__init__(self, **kwargs)
        assert self.condition_key in ['center', 'bbox']  # for annotated images only

    @staticmethod
    def parseOpencvMatrix(node):
        rows = int(node.find('rows').text)
        cols = int(node.find('cols').text)
        data = node.find('data').text.split(' ')

        mat = []
        for d in data:
            d = d.replace('\n', '')
            if len(d) < 1:
                continue
            mat.append(float(d))
        mat = np.reshape(mat, [rows, cols])
        return mat

    def parseVertices(self, child):
        transform = self.parseOpencvMatrix(child.find('transform'))
        R = transform[:3, :3]
        T = transform[:3, 3]
        vertices = self.parseOpencvMatrix(child.find('vertices'))
        vertices = np.matmul(R, vertices.transpose()).transpose() + T
        return vertices

    def parse_bbox_xml(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        bbox_dict = dict()
        label_dict = dict()
        for child in root:
            if child.find('transform') is None:
                continue

            label_name = child.find('label').text
            if label_name not in BBOX_CAT2LABEL:
                continue

            label = BBOX_CAT2LABEL[label_name]
            timestamp = int(child.find('timestamp').text)
            # verts = self.parseVertices(child)
            verts = self.parseOpencvMatrix(child.find('vertices'))[:8]
            if timestamp in bbox_dict:
                bbox_dict[timestamp].append(verts)
                label_dict[timestamp].append(label)
            else:
                bbox_dict[timestamp] = [verts]
                label_dict[timestamp] = [label]
        return bbox_dict, label_dict

    def prepare_data(self):
        KITTI360Base.prepare_data(self)

        self.data = [p for p in self.data if '2013_05_28_drive_0008_sync' not in p]  # remove unlabeled sequence 08
        seq_list = eval('KITTI360_%s_SET' % self.split.upper())
        for seq_id in seq_list:
            if seq_id != '08':
                xml_path = os.path.join(self.data_root, f'data_3d_bboxes/train/2013_05_28_drive_00{seq_id}_sync.xml')
                bbox_dict, label_dict = self.parse_bbox_xml(xml_path)
                self.id_bbox_dict[seq_id] = bbox_dict
                self.id_label_dict[seq_id] = label_dict

    def load_annotation(self, path):
        seq_id = path.split('/')[-4].split('_')[-2][-2:]
        timestamp = int(path.split('/')[-1].replace('.bin', ''))
        verts_list = self.id_bbox_dict[seq_id][timestamp]
        label_list = self.id_label_dict[seq_id][timestamp]

        if self.condition_key == 'bbox':
            points = np.stack(verts_list)
        elif self.condition_key == 'center':
            points = (verts_list[0] + verts_list[6]) / 2.
        else:
            raise NotImplementedError
        labels = np.array([label_list])
        if self.anno_transform:
            points, labels = self.anno_transform(points, labels)
        return points, labels

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]

        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        # annotations
        bbox_points, bbox_labels = self.load_annotation(data_path)

        if self.lidar_transform:
            sweep, bbox_points = self.lidar_transform(sweep, bbox_points)

        # point cloud -> range
        proj_range, _ = pcd2range(sweep, self.img_size, self.fov, self.depth_range)
        proj_range, proj_mask = self.process_scan(proj_range)
        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            example['reproj'] = sweep

        # annotation -> range
        # NOTE: do not need to transform bbox points along with lidar, since their coordinates are based on range-image space instead of 3D space
        proj_bbox_points, proj_bbox_labels = pcd2coord2d(bbox_points, self.fov, self.depth_range, labels=bbox_labels)
        builder = self.conditional_builders[self.condition_key]
        if self.condition_key == 'bbox':
            proj_bbox_points = corners_3d_to_2d(proj_bbox_points)
            annotations = [Annotation(bbox=bbox.flatten(), category_id=label) for bbox, label in
                           zip(proj_bbox_points, proj_bbox_labels)]
        else:
            annotations = [Annotation(center=center, category_id=label) for center, label in
                           zip(proj_bbox_points, proj_bbox_labels)]
        example[self.condition_key] = builder.build(annotations)

        return example


class AnnotatedKITTI360Train(AnnotatedKITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='train', cats=BBOX_CATS, **kwargs)


class AnnotatedKITTI360Validation(AnnotatedKITTI360Base):
    def __init__(self, **kwargs):
        super().__init__(data_root='./dataset/KITTI-360', split='train', cats=BBOX_CATS, **kwargs)


class KITTIImageBase(KITTIBase):
    """
    Range ImageSet only combining KITTI-360 and SemanticKITTI

    #Samples (Training): 98014, #Samples (Val): 3511

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.condition_key in [None, 'image']  # for image input only

    def prepare_data(self):
        # read data paths from KITTI-360
        self.data = []
        for seq_id in eval('KITTI360_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'KITTI-360/data_3d_raw/2013_05_28_drive_00{seq_id}_sync/velodyne_points/data/*.bin')))

        # read data paths from KITTI
        for seq_id in eval('KITTI_%s_SET' % self.split.upper()):
            self.data.extend(glob.glob(os.path.join(
                self.data_root, f'SemanticKITTI/dataset/sequences/{seq_id}/velodyne/*.bin')))


class KITTIImageTrain(KITTIImageBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/', split='train', **kwargs)


class KITTIImageValidation(KITTIImageBase):
    def __init__(self, **kwargs):
        super().__init__(data_root='/data1/dataset/', split='val', **kwargs)


#! 添加waymo数据集 

class HDF5Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            pkl_directory, 
            data_path, 
            tokenizer=None,
            n_frames=2, 
            eval = False, 
            height=64,
            width=1024, # resize
            depth_max=80.0,          # depth 上限（用于归一化）
            crop_w_px=0,            # 宽度两侧各裁多少像素
            crop_w_both=True,        # True=两侧各裁；False=只裁右侧
            mask_method="hard", 
            sigma=0.12, 
            eps=0.03,
            ):
        '''
        Args:
            directory: the directory of the dataset, the folder should contain some .h5 file and index_total.pkl.

            Following are optional:
            * n_frames: the number of frames we use, default is 2: from pc0 to pc1.
            * ssl_label: if not None, we will use this label for self-supervised learning
            * eval: if True, use the eval index
            * leaderboard_version: 1st or 2nd, default is 1. If '2', we will use the index_eval_v2.pkl from assets/docs.
        '''
        super(HDF5Dataset, self).__init__()
        self.pkl_directory = pkl_directory
        self.data_path = data_path
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, eval={eval}")
        #! waymo数据集处理
        with open(os.path.join(self.pkl_directory, 'train_500.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)

            # self.data_index = sorted(self.data_index)[0:1]
        #finetune oneseq Difix3D-main/assets/segment-1887497421568128425_94_000_114_000_with_camera_labels_train_6.pkl
        self.eval_index = False

        self.H = int(height)
        self.W = int(width)
        self.tokenizer = tokenizer

        self.depth_max = float(depth_max)
        self.crop_w_px = int(crop_w_px)
        self.crop_w_both = bool(crop_w_both)

        self.mask_method = str(mask_method)
        self.sigma = float(sigma)
        self.eps = float(eps)

        if eval:
            eval_index_file = os.path.join(self.pkl_directory, 'index_lidarrt_val.pkl') #! 与lidar-rt保持一致 waymo

            #和lidar-rt保持一致的pkl index_lidarrt_val.pkl #这个是lidar-rt完全一致的val集，只有32帧
            #test_lidar-rt.pkl 是lidar-rt整个序列都作为验证集。
            #lidar-rt+额外的数据 val_28.pkl
            #finetune oneseq segment-1887497421568128425_94_000_114_000_with_camera_labels_val_7.pkl
            #'test_300_400.pkl'
            if not os.path.exists(eval_index_file):
                raise Exception(f"No eval index file found! Please check {self.pkl_directory}")
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)
                self.eval_data_index = sorted(self.eval_data_index)
                # print(self.eval_data_index)
                # self.eval_data_index = sorted(self.eval_data_index)[0:1]


    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    


    # ---------- 基础工具 ----------
    @staticmethod
    def _to_chw_float32(arr):
        # arr: HxWxC -> tensor [C,H,W], float32 (不改变数值范围)
        return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)

    @staticmethod
    def _resize_chw(t, H, W, mode='bilinear', align_corners=False):
        # t: [C,h,w] -> [C,H,W]
        if mode == 'nearest':
            align_corners = None
        return F.interpolate(t.unsqueeze(0), size=(H, W), mode=mode, align_corners=align_corners).squeeze(0)

    @staticmethod
    def _clip01(x):
        return torch.clamp(x, 0.0, 1.0)

    def _crop_w_edges(self, t, px=10, both=True):
        """
        t: [C,H,W]
        both=True  -> 从左右各裁掉 px
        both=False -> 只从右边裁掉 px
        """
        if px <= 0:
            return t
        C, H, W = t.shape
        if both:
            if W <= 2 * px:
                raise ValueError(f"Width {W} too small to crop both sides {px} each.")
            return t[..., px: W - px]
        else:
            if W <= px:
                raise ValueError(f"Width {W} too small to crop right side {px}.")
            return t[..., : W - px]


    @torch.no_grad()
    def depth_to_dropmask(self,
        depth_norm: torch.Tensor,
        method: str = "soft",   # "soft" 高斯先验；"hard" 阈值
        sigma: float = 0.12,    # soft: 距离(-1)的尺度(归一化域)
        eps: float = 0.05,      # hard: 与 -1 的距离阈值
        clamp01: bool = True,
    ) -> torch.Tensor:
        """
        将归一化 depth([-1,1]) 映射为 drop 先验 mask∈[0,1]。
        你定义的是：越接近 -1 越可能是 drop（0米回波）。
        - soft:  m = exp(- ((depth_norm + 1)/sigma)^2)    # d≈-1 -> m≈1
        - hard:  m = 1[ |depth_norm + 1| < eps ]
        """
        assert depth_norm.is_floating_point(), "depth_norm must be float"
        d = depth_norm
        if d.ndim == 2:  # (H,W) -> (1,H,W)
            d = d.unsqueeze(0)

        # 与 -1 的“距离”
        dist = (d + 1.0).abs()   # d=-1 时 dist=0（最可能 drop）

        if method == "soft":
            m = torch.exp(- (dist / max(sigma, 1e-6)) ** 2)
        elif method == "hard":
            m = (dist < eps).to(d.dtype)
        else:
            raise ValueError(f"Unknown method: {method}")

        if clamp01:
            m = m.clamp_(0.0, 1.0)
        return m.float()



    # ---------- 归一化/反归一 ----------
    def _norm_2ch_to_m11(self, t_2ch):
        """
        t_2ch: [2,H,W], ch0=depth(0~depth_max), ch1=intensity(0~1)
        输出: [-1,1]
        """
        depth = t_2ch[0:1] / self.depth_max       # -> [0,1]
        depth = self._clip01(depth)
        assert t_2ch[1:2].max() < 1.1
        inten = self._clip01(t_2ch[1:2])          # 已是[0,1]范围假设

        depth = depth * 2.0 - 1.0                 # -> [-1,1]
        inten = inten * 2.0 - 1.0
        return torch.cat([depth, inten], dim=0)   # [2,H,W]

    def _denorm_2ch_from_m11(self, t_2ch_norm):
        """
        可用于可视化：把 [-1,1] 还原回物理量（depth回到米，inten回到0~1）
        """
        depth = (t_2ch_norm[0:1] + 1.0) * 0.5 * self.depth_max
        inten = (t_2ch_norm[1:2] + 1.0) * 0.5
        depth = torch.clamp(depth, 0.0, self.depth_max)
        inten = torch.clamp(inten, 0.0, 1.0)
        return torch.cat([depth, inten], dim=0)

    # ---------- 读取并处理 ----------
    def _load_input_3ch(self, path_hw2):
        """
        image/ref: HxWx2 -> [3,H,W]
        (前两通道归一到[-1,1]，第三通道是占位常数0)
        """
        arr = path_hw2
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(f"Expect HxWx2 npy, got {arr.shape}")

        t2 = self._to_chw_float32(arr)                        # [2,h,w]
        t2 = self._resize_chw(t2, self.H, self.W, 'bilinear', False)   # [2,H,W]
        t2 = self._norm_2ch_to_m11(t2)                        # -> [-1,1]

        dummy = torch.zeros((1, self.H, self.W), dtype=t2.dtype, device=t2.device)  # 占位通道（供网络学习）
        depth_norm = t2[0:1, :, :] 
        drop_prior = self.depth_to_dropmask(depth_norm, method=self.mask_method, sigma=self.sigma, eps=self.eps)  # [1,H,W]  drop=1说明被丢弃  
        t3 = torch.cat([t2, drop_prior], dim=0)                    # [3,H,W]

        # 宽度边缘裁掉
        t3 = self._crop_w_edges(t3, px=self.crop_w_px, both=self.crop_w_both)       # [3,H,Wc]
        return t3

    def _load_target_3ch(self, path_hw3):
        """
        target_image: HxWx3 -> [3,H,W]
        - 第0/1通道（d/i）归一到[-1,1]
        - 第2通道保持原始数值（如需也归一，可在此处自行修改）
        """

        arr = path_hw3
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expect HxWx3 npy, got {arr.shape}")

        t3 = self._to_chw_float32(arr)                        # [3,h,w]

        t01 = self._norm_2ch_to_m11(t3[:2])                   # [2,H,W]
        t2  = t3[2:3]                                         # [1,H,W] 保持原样
        out = torch.cat([t01, t2], dim=0)                     # [3,H,W]
        out = self._crop_w_edges(out, px=self.crop_w_px, both=self.crop_w_both)      # [3,H,Wc]
        #! resize到指定大小
        t3_resize = self._resize_chw(t3[:2], self.H, self.W, 'bilinear', False)   # [3,H,W]

        # 只对 d/i 做 [-1,1]
        t01_resize = self._norm_2ch_to_m11(t3_resize[:2])                   # [2,H,W]
        t2_resize  = self._resize_chw(t3[2:3], self.H, self.W, mode='nearest')                                         # [1,H,W] 保持原样
        out_resize = torch.cat([t01_resize, t2_resize], dim=0)                     # [3,H,W]

        # 宽度边缘裁掉
        out_resize = self._crop_w_edges(out_resize, px=self.crop_w_px, both=self.crop_w_both)      # [3,H,Wc]



        
        return out_resize, out


    def __getitem__(self, index_):
        # if self.eval_index:
        #     scene_id, timestamp, timestamp_prev = self.eval_data_index[index_]
        # else:
        #     scene_id, timestamp, timestamp_prev = self.data_index[index_]

        if self.eval_index:
            scene_id, timestamp = self.eval_data_index[index_]
        else:
            scene_id, timestamp = self.data_index[index_]



        key = str(timestamp)
        # key_prev = str(timestamp_prev)

        with h5py.File(os.path.join(self.data_path, f'{scene_id}.h5'), 'r') as f:
            g = f[key]
            # g_pre = f[key_prev]
            pano_cat =  g["pano_cat_return1_2"][...]   #6通道, 0depth，1intensity, 2-4normals, 5labels3D.    # (H,W,C)，C>=? 约定[...,0]为深度
            gt_cat = g["gt_cat"][...]                    # (H,W,3): depth,intensity,mask
            # gt_cat_prev = g_pre["gt_cat"][...]            # (H,W,3): depth,intensity,mask
            ir = g["ir"][...]                            # (H,) 或 (2,)
            sensor_center = g["sensor_center"][...]      # (3,)
            T_w = g["sensor2world"][...]                 # (4,4)
            angle_offset = float(g.attrs["angle_offset"])
            pixel_offset = float(g.attrs["pixel_offset"])
            caption = 'remove degradation'  #!固定为原始的remove degradation
            #! 将数据normalize到[-1,1] # h,w,2-->3,h,w 除去max_depth=80
            # 填补dropout通道为0，三个通道归一化到（-1,1）
            img_t =  self._load_input_3ch(pano_cat[:,:,:2]) # [3,h,w]
            output_t_resize, output_t = self._load_target_3ch(gt_cat[:,:,:3])   # [3,H,Wc]  
            # output_t_prev = self._load_target_3ch(gt_cat_prev[:,:,:3])   # [3,H,Wc] 
            # img_t[2:3,:,:] = 1.0 - img_t[2:3,:,:]
            # img_t[2:3,:,:] = 1.0 - ((output_t_prev[2:3,:,:]==1) & (img_t[2:3,:,:]==0)).float() #! dropout
            # img_t[2:3,:,:] = 1.0 - ((output_t_prev[2:3,:,:]==1)).float() #! dropout


        ref_img = None #! 目前没有ref_img
        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t
            output_t = output_t
            output_t_resize = output_t_resize

        out = {
            "image": output_t_resize[:1], #label resize之后
            "image_ori": output_t[:1], #label原始大小
            "mask_ori": output_t[2:3], #原始mask大小
            "segmentation": img_t[:1], #累积之后的rangeimage作为condition
            "caption": caption,
            'ir': ir,
            'sensor_center': sensor_center, # [3]
            'sensor2world':  T_w,  # [4,4]
            'angle_offset':  angle_offset,  # 标量
            'pixel_offset':  pixel_offset,  # 标量
            'scene_id': scene_id,
            'timestamp': timestamp,
        }


        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids


        return out
    


class HDF5Dataset_kitti(torch.utils.data.Dataset):
    def __init__(
            self, 
            pkl_directory, 
            data_path, 
            tokenizer=None,
            n_frames=2, 
            eval = False, 
            height=64,
            width=1024, # resize
            depth_max=80.0,          # depth 上限（用于归一化）
            crop_w_px=0,            # 宽度两侧各裁多少像素
            crop_w_both=True,        # True=两侧各裁；False=只裁右侧
            mask_method="hard", 
            sigma=0.12, 
            eps=0.03,
            ):
        '''
        Args:
            directory: the directory of the dataset, the folder should contain some .h5 file and index_total.pkl.

            Following are optional:
            * n_frames: the number of frames we use, default is 2: from pc0 to pc1.
            * ssl_label: if not None, we will use this label for self-supervised learning
            * eval: if True, use the eval index
            * leaderboard_version: 1st or 2nd, default is 1. If '2', we will use the index_eval_v2.pkl from assets/docs.
        '''
        super(HDF5Dataset_kitti, self).__init__()
        self.pkl_directory = pkl_directory
        self.data_path = data_path
        if (torch.distributed.is_initialized() and torch.distributed.get_rank() == 0) or not torch.distributed.is_initialized():
            print(f"----[Debug] Loading data with num_frames={n_frames}, eval={eval}")
        #! waymo数据集处理
        # with open(os.path.join(self.pkl_directory, 'train_500.pkl'), 'rb') as f:
        #     self.data_index = pickle.load(f)
        #! if kitti
        with open(os.path.join(self.pkl_directory, 'index_train_kitti.pkl'), 'rb') as f:
            self.data_index = pickle.load(f)
            # self.data_index = sorted(self.data_index)[0:1]
        #finetune oneseq Difix3D-main/assets/segment-1887497421568128425_94_000_114_000_with_camera_labels_train_6.pkl
        self.eval_index = False

        self.H = int(height)
        self.W = int(width)
        self.tokenizer = tokenizer

        self.depth_max = float(depth_max)
        self.crop_w_px = int(crop_w_px)
        self.crop_w_both = bool(crop_w_both)

        self.mask_method = str(mask_method)
        self.sigma = float(sigma)
        self.eps = float(eps)

        if eval:
            # eval_index_file = os.path.join(self.pkl_directory, 'index_lidarrt_val.pkl') #! 与lidar-rt保持一致 waymo
            eval_index_file = os.path.join(self.pkl_directory, 'index_test_kitti.pkl') #! kitti
            #和lidar-rt保持一致的pkl index_lidarrt_val.pkl #这个是lidar-rt完全一致的val集，只有32帧
            #test_lidar-rt.pkl 是lidar-rt整个序列都作为验证集。
            #lidar-rt+额外的数据 val_28.pkl
            #finetune oneseq segment-1887497421568128425_94_000_114_000_with_camera_labels_val_7.pkl
            #'test_300_400.pkl'
            if not os.path.exists(eval_index_file):
                raise Exception(f"No eval index file found! Please check {self.pkl_directory}")
            self.eval_index = eval
            with open(eval_index_file, 'rb') as f:
                self.eval_data_index = pickle.load(f)
                self.eval_data_index = sorted(self.eval_data_index)
                # print(self.eval_data_index)
                # self.eval_data_index = sorted(self.eval_data_index)[0:1]


    def __len__(self):
        if self.eval_index:
            return len(self.eval_data_index)
        return len(self.data_index)
    


    # ---------- 基础工具 ----------
    @staticmethod
    def _to_chw_float32(arr):
        # arr: HxWxC -> tensor [C,H,W], float32 (不改变数值范围)
        return torch.from_numpy(arr.astype(np.float32)).permute(2, 0, 1)

    @staticmethod
    def _resize_chw(t, H, W, mode='bilinear', align_corners=False):
        # t: [C,h,w] -> [C,H,W]
        if mode == 'nearest':
            align_corners = None
        return F.interpolate(t.unsqueeze(0), size=(H, W), mode=mode, align_corners=align_corners).squeeze(0)

    @staticmethod
    def _clip01(x):
        return torch.clamp(x, 0.0, 1.0)

    def _crop_w_edges(self, t, px=10, both=True):
        """
        t: [C,H,W]
        both=True  -> 从左右各裁掉 px
        both=False -> 只从右边裁掉 px
        """
        if px <= 0:
            return t
        C, H, W = t.shape
        if both:
            if W <= 2 * px:
                raise ValueError(f"Width {W} too small to crop both sides {px} each.")
            return t[..., px: W - px]
        else:
            if W <= px:
                raise ValueError(f"Width {W} too small to crop right side {px}.")
            return t[..., : W - px]


    @torch.no_grad()
    def depth_to_dropmask(self,
        depth_norm: torch.Tensor,
        method: str = "soft",   # "soft" 高斯先验；"hard" 阈值
        sigma: float = 0.12,    # soft: 距离(-1)的尺度(归一化域)
        eps: float = 0.05,      # hard: 与 -1 的距离阈值
        clamp01: bool = True,
    ) -> torch.Tensor:
        """
        将归一化 depth([-1,1]) 映射为 drop 先验 mask∈[0,1]。
        你定义的是：越接近 -1 越可能是 drop（0米回波）。
        - soft:  m = exp(- ((depth_norm + 1)/sigma)^2)    # d≈-1 -> m≈1
        - hard:  m = 1[ |depth_norm + 1| < eps ]
        """
        assert depth_norm.is_floating_point(), "depth_norm must be float"
        d = depth_norm
        if d.ndim == 2:  # (H,W) -> (1,H,W)
            d = d.unsqueeze(0)

        # 与 -1 的“距离”
        dist = (d + 1.0).abs()   # d=-1 时 dist=0（最可能 drop）

        if method == "soft":
            m = torch.exp(- (dist / max(sigma, 1e-6)) ** 2)
        elif method == "hard":
            m = (dist < eps).to(d.dtype)
        else:
            raise ValueError(f"Unknown method: {method}")

        if clamp01:
            m = m.clamp_(0.0, 1.0)
        return m.float()



    # ---------- 归一化/反归一 ----------
    def _norm_2ch_to_m11(self, t_2ch):
        """
        t_2ch: [2,H,W], ch0=depth(0~depth_max), ch1=intensity(0~1)
        输出: [-1,1]
        """
        depth = t_2ch[0:1] / self.depth_max       # -> [0,1]
        depth = self._clip01(depth)
        assert t_2ch[1:2].max() < 1.1
        inten = self._clip01(t_2ch[1:2])          # 已是[0,1]范围假设

        depth = depth * 2.0 - 1.0                 # -> [-1,1]
        inten = inten * 2.0 - 1.0
        return torch.cat([depth, inten], dim=0)   # [2,H,W]

    def _denorm_2ch_from_m11(self, t_2ch_norm):
        """
        可用于可视化：把 [-1,1] 还原回物理量（depth回到米，inten回到0~1）
        """
        depth = (t_2ch_norm[0:1] + 1.0) * 0.5 * self.depth_max
        inten = (t_2ch_norm[1:2] + 1.0) * 0.5
        depth = torch.clamp(depth, 0.0, self.depth_max)
        inten = torch.clamp(inten, 0.0, 1.0)
        return torch.cat([depth, inten], dim=0)

    # ---------- 读取并处理 ----------
    def _load_input_3ch(self, path_hw2):
        """
        image/ref: HxWx2 -> [3,H,W]
        (前两通道归一到[-1,1]，第三通道是占位常数0)
        """
        arr = path_hw2
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(f"Expect HxWx2 npy, got {arr.shape}")

        t2 = self._to_chw_float32(arr)                        # [2,h,w]
        # t2 = self._resize_chw(t2, self.H, self.W, 'bilinear', False)   # [2,H,W] #不resize, 直接cut数据
        t2 = t2[:, 1:-1, 3:-3]
        t2 = self._norm_2ch_to_m11(t2)                        # -> [-1,1]

        dummy = torch.zeros((1, self.H, self.W), dtype=t2.dtype, device=t2.device)  # 占位通道（供网络学习）
        depth_norm = t2[0:1, :, :] 
        drop_prior = self.depth_to_dropmask(depth_norm, method=self.mask_method, sigma=self.sigma, eps=self.eps)  # [1,H,W]  drop=1说明被丢弃  
        t3 = torch.cat([t2, drop_prior], dim=0)                    # [3,H,W]

        # 宽度边缘裁掉
        t3 = self._crop_w_edges(t3, px=self.crop_w_px, both=self.crop_w_both)       # [3,H,Wc]
        return t3

    def _load_target_3ch(self, path_hw3):
        """
        target_image: HxWx3 -> [3,H,W]
        - 第0/1通道（d/i）归一到[-1,1]
        - 第2通道保持原始数值（如需也归一，可在此处自行修改）
        """

        arr = path_hw3
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"Expect HxWx3 npy, got {arr.shape}")

        t3 = self._to_chw_float32(arr)                        # [3,h,w]

        t01 = self._norm_2ch_to_m11(t3[:2])                   # [2,H,W]
        t2  = t3[2:3]                                         # [1,H,W] 保持原样
        out = torch.cat([t01, t2], dim=0)                     # [3,H,W]
        out = self._crop_w_edges(out, px=self.crop_w_px, both=self.crop_w_both)      # [3,H,Wc]
        #! resize到指定大小
        t3_resize = self._resize_chw(t3[:2], self.H, self.W, 'bilinear', False)   # [3,H,W]

        # 只对 d/i 做 [-1,1]
        t01_resize = self._norm_2ch_to_m11(t3_resize[:2])                   # [2,H,W]
        t2_resize  = self._resize_chw(t3[2:3], self.H, self.W, mode='nearest')                                         # [1,H,W] 保持原样
        out_resize = torch.cat([t01_resize, t2_resize], dim=0)                     # [3,H,W]

        # 宽度边缘裁掉
        out_resize = self._crop_w_edges(out_resize, px=self.crop_w_px, both=self.crop_w_both)      # [3,H,Wc]



        
        return out_resize, out


    def __getitem__(self, index_):
        # if self.eval_index:
        #     scene_id, timestamp, timestamp_prev = self.eval_data_index[index_]
        # else:
        #     scene_id, timestamp, timestamp_prev = self.data_index[index_]

        if self.eval_index:
            seq, scene_id, timestamp = self.eval_data_index[index_]
        else:
            seq, scene_id, timestamp = self.data_index[index_]



        key = str(timestamp)
        # key_prev = str(timestamp_prev)

        with h5py.File(os.path.join(self.data_path, seq, f'{scene_id}.h5'), 'r') as f:
            g = f[key]
            # g_pre = f[key_prev]
            pano_cat =  g["pano_cat_return1_2"][...]   #6通道, 0depth，1intensity, 2-4normals, 5labels3D.    # (H,W,C)，C>=? 约定[...,0]为深度
            gt_cat = g["gt_cat"][...]                    # (H,W,3): depth,intensity,mask
            # gt_cat_prev = g_pre["gt_cat"][...]            # (H,W,3): depth,intensity,mask
            # ir = g["ir"][...]                            # (H,) 或 (2,)
            sensor_center = g["sensor_center"][...]      # (3,)
            T_w = g["sensor2world"][...]
            if T_w.shape == (4,4):
                 T_w = T_w[:3,:] #!      
            # angle_offset = float(g.attrs["angle_offset"])
            # pixel_offset = float(g.attrs["pixel_offset"])
            caption = 'remove degradation'  #!固定为原始的remove degradation
            #! 将数据normalize到[-1,1] # h,w,2-->3,h,w 除去max_depth=80
            # 填补dropout通道为0，三个通道归一化到（-1,1）
            img_t =  self._load_input_3ch(pano_cat[:,:,:2]) # [3,h,w]
            output_t_resize, output_t = self._load_target_3ch(gt_cat[:,:,:3])   # [3,H,Wc]  
            # output_t_prev = self._load_target_3ch(gt_cat_prev[:,:,:3])   # [3,H,Wc] 
            # img_t[2:3,:,:] = 1.0 - img_t[2:3,:,:]
            # img_t[2:3,:,:] = 1.0 - ((output_t_prev[2:3,:,:]==1) & (img_t[2:3,:,:]==0)).float() #! dropout
            # img_t[2:3,:,:] = 1.0 - ((output_t_prev[2:3,:,:]==1)).float() #! dropout


        ref_img = None #! 目前没有ref_img
        if ref_img is not None:
            ref_img = Image.open(ref_img)
            ref_t = F.to_tensor(ref_t)
            ref_t = F.resize(ref_t, self.image_size)
            ref_t = F.normalize(ref_t, mean=[0.5], std=[0.5])
        
            img_t = torch.stack([img_t, ref_t], dim=0)
            output_t = torch.stack([output_t, ref_t], dim=0)            
        else:
            img_t = img_t
            output_t = output_t
            output_t_resize = output_t_resize

        out = {
            "image": output_t_resize[:1], #label resize之后
            "image_ori": output_t[:1], #label原始大小
            "mask_ori": output_t[2:3], #原始mask大小
            "segmentation": img_t[:1], #累积之后的rangeimage作为condition
            "caption": caption,
            # 'ir': ir, 
            'sensor_center': sensor_center, # [3]
            'sensor2world':  T_w,  # [4,4]
            # 'angle_offset':  angle_offset,  # 标量
            # 'pixel_offset':  pixel_offset,  # 标量
            'scene_id': scene_id,
            'timestamp': timestamp,
        }


        if self.tokenizer is not None:
            input_ids = self.tokenizer(
                caption, max_length=self.tokenizer.model_max_length,
                padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids
            out["input_ids"] = input_ids


        return out