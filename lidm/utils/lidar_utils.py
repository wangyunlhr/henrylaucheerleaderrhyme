import math

import numpy as np


def pcd2coord2d(pcd, fov, depth_range, labels=None):
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=-1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    if pcd.ndim == 3:
        mask = mask.all(axis=1)
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[..., 0], pcd[..., 1], pcd[..., 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = np.clip(0.5 * (yaw / np.pi + 1.0), 0., 1.)  # in [0.0, 1.0]
    proj_y = np.clip(1.0 - (pitch + abs(fov_down)) / fov_range, 0., 1.)  # in [0.0, 1.0]
    proj_coord2d = np.stack([proj_x, proj_y], axis=-1)

    if labels is not None:
        proj_labels = labels[mask]
    else:
        proj_labels = None

    return proj_coord2d, proj_labels

#! lidar-rt代码
def kitti_points_to_range_image(xyzs,
                          intensities,
                          H=66,
                          W=1030,
                          inc_bottom=math.radians(-24.9),
                          inc_top=math.radians(2.0),
                          azimuth_left=np.pi,
                          azimuth_right=-np.pi,
                          max_depth=80.0):
    """
    将 LiDAR 点云投影为 range image + intensity image

    Args:
        xyzs:        (N, 3) 点坐标（在 LiDAR 坐标系）
        intensities: (N,)   对应每个点的强度
        H, W:        range image 的高度和宽度
        inc_bottom:  垂直下边界（弧度）
        inc_top:     垂直上边界（弧度）
        azimuth_left, azimuth_right: 水平角范围（左/右，弧度）
        max_depth:   最大深度，超过则丢弃

    Returns:
        range_map:      (H, W) 距离图
        intensity_map:  (H, W) 强度图
    """

    # 水平/垂直角分辨率
    h_res = (azimuth_right - azimuth_left) / W
    v_res = (inc_bottom - inc_top) / H

    # 初始化为 -1，表示空
    C = intensities.shape[1]
    range_map = np.ones((H, W), dtype=np.float32) * -1
    intensity_map = np.ones((H, W, C), dtype=np.float32) * -1

    # 预计算距离
    dists = np.linalg.norm(xyzs, axis=1)

    for xyz, intensity, dist in zip(xyzs, intensities, dists):
        x, y, z = xyz

        # 计算水平角 azimuth 和俯仰角 inclination
        azimuth = np.arctan2(y, x)
        inclination = np.arctan2(z, np.sqrt(x**2 + y**2))

        # 深度过滤
        if dist > max_depth:
            continue

        # 投影到像素坐标
        w_idx = np.round((azimuth - azimuth_left) / h_res).astype(int)
        h_idx = np.round((inclination - inc_top) / v_res).astype(int)

        # 越界丢弃
        if (w_idx < 0) or (w_idx >= W) or (h_idx < 0) or (h_idx >= H):
            continue

        # 只保留最近的点
        if range_map[h_idx, w_idx] == -1 or range_map[h_idx, w_idx] > dist:
            range_map[h_idx, w_idx] = dist
            intensity_map[h_idx, w_idx] = intensity

    range_map_3d = range_map[..., None]
    range_image_r1 = np.concatenate([range_map_3d, intensity_map], axis=-1)  
    range_image_r1[range_map == -1] = 0
    
    return range_image_r1

def pcd2range(pcd, size, fov, depth_range, remission=None, labels=None, **kwargs):
    # laser parameters
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth (distance) of all points
    depth = np.linalg.norm(pcd, 2, axis=1)

    # mask points out of range
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    depth, pcd = depth[mask], pcd[mask]

    # get scan components
    scan_x, scan_y, scan_z = pcd[:, 0], pcd[:, 1], pcd[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov_range  # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= size[1]  # in [0.0, W]
    proj_y *= size[0]  # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.maximum(0, np.minimum(size[1] - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(size[0] - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    proj_x, proj_y = proj_x[order], proj_y[order]

    # project depth
    depth = depth[order]
    proj_range = np.full(size, -1, dtype=np.float32)
    proj_range[proj_y, proj_x] = depth

    # project point feature
    if remission is not None:
        remission = remission[mask][order]
        proj_feature = np.full(size, -1, dtype=np.float32)
        proj_feature[proj_y, proj_x] = remission
    elif labels is not None:
        labels = labels[mask][order]
        proj_feature = np.full(size, 0, dtype=np.float32)
        proj_feature[proj_y, proj_x] = labels
    else:
        proj_feature = None

    return proj_range, proj_feature


def range2pcd(range_img, fov, depth_range, depth_scale, log_scale=True, label=None, color=None, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    depth = (range_img * depth_scale).flatten()
    if log_scale:
        depth = np.exp2(depth) - 1

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = (np.pi * (scan_x * 2 - 1)).flatten()
    pitch = ((1.0 - scan_y) * fov_range - abs(fov_down)).flatten()

    pcd = np.zeros((len(yaw), 3))
    pcd[:, 0] = np.cos(yaw) * np.cos(pitch) * depth
    pcd[:, 1] = -np.sin(yaw) * np.cos(pitch) * depth
    pcd[:, 2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    pcd = pcd[mask, :]

    # label
    if label is not None:
        label = label.flatten()[mask]

    # default point color
    if color is not None:
        color = color.reshape(-1, 3)[mask, :]
    else:
        color = np.ones((pcd.shape[0], 3)) * [0.7, 0.7, 1]

    return pcd, color, label


def range2xyz(range_img, fov, depth_range, depth_scale, log_scale=True, **kwargs):
    # laser parameters
    size = range_img.shape
    fov_up = fov[0] / 180.0 * np.pi  # field of view up in rad
    fov_down = fov[1] / 180.0 * np.pi  # field of view down in rad
    fov_range = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # inverse transform from depth
    if log_scale:
        depth = (np.exp2(range_img * depth_scale) - 1)
    else:
        depth = range_img

    scan_x, scan_y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    scan_x = scan_x.astype(np.float64) / size[1]
    scan_y = scan_y.astype(np.float64) / size[0]

    yaw = np.pi * (scan_x * 2 - 1)
    pitch = (1.0 - scan_y) * fov_range - abs(fov_down)

    xyz = -np.ones((3, *size))
    xyz[0] = np.cos(yaw) * np.cos(pitch) * depth
    xyz[1] = -np.sin(yaw) * np.cos(pitch) * depth
    xyz[2] = np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth > depth_range[0], depth < depth_range[1])
    xyz[:, ~mask] = -1

    return xyz


def pcd2bev(pcd, x_range, y_range, z_range, resolution, **kwargs):
    # mask out invalid points
    mask_x = np.logical_and(pcd[:, 0] > x_range[0], pcd[:, 0] < x_range[1])
    mask_y = np.logical_and(pcd[:, 1] > y_range[0], pcd[:, 1] < y_range[1])
    mask_z = np.logical_and(pcd[:, 2] > z_range[0], pcd[:, 2] < z_range[1])
    mask = mask_x & mask_y & mask_z
    pcd = pcd[mask]

    # points to bev coords
    bev_x = np.floor((pcd[:, 0] - x_range[0]) / resolution).astype(np.int32)
    bev_y = np.floor((pcd[:, 1] - y_range[0]) / resolution).astype(np.int32)

    # 2D bev grid
    bev_shape = (math.ceil((x_range[1] - x_range[0]) // resolution), math.ceil((y_range[1] - y_range[0]) // resolution))
    bev_grid = np.zeros(bev_shape, dtype=np.float64)

    # populate the BEV grid with bev coords
    bev_grid[bev_x, bev_y] = 1

    return bev_grid


if __name__ == '__main__':
    # test = np.loadtxt('test_range.txt')
    # pcd, _, _ = range2pcd(test, (32, 1024), (10, -30))
    # np.savetxt('test_pcd.txt', pcd, fmt='%.4f')

    # import matplotlib.pyplot as plt
    # pcd = np.loadtxt('test_origin.txt')
    # bev_grid = pcd2bev(pcd)
    # plt.imshow(bev_grid[:, :, 0], cmap='gray')  # Display the BEV for the first height level
    # plt.savefig('test.png', dpi=300, bbox_inches='tight', pad_inches=0, transparent=True)

    from PIL import Image
    img = Image.open('assets/kitti/range.png')
    img.convert('L')
    img = np.array(img) / 255.
