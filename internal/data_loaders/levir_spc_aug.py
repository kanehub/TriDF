import os
import torch
from torch.utils.data import Dataset
import numpy as np
from plyfile import PlyData
from PIL import Image
import cv2

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from internal.data_loaders import LevirNVSDataset

class LevirSPCAugDataset(Dataset):
    '''
    # augmentation point cloud dataset
    '''
    def __init__(self,
            args,
            ray_dataset: LevirNVSDataset,
            device,
            is_augmentation,
            select_single_view_pts,
            resample=False
        ):

        self.args = args
        self.file_path = args.data_path + r'/LEVIR_SPC/' + args.scene_id
        self.ray_dataset = ray_dataset
        self.device = device
        self.resample = resample
        self.augmentation = is_augmentation
        self.select_single_view_pts = select_single_view_pts

        # aug_path = os.path.join(self.file_path, 'aug_points3D.npy')
        # if self.resample or not os.path.exists(aug_path):
        #     aug_point_cloud = self.generate_spc()
        # else:
        #     aug_point_cloud = np.load(aug_path)
        ply_path = os.path.join(self.file_path, 'dense.ply')
        ori_point_cloud, ori_point_rgb = self.load_ply(ply_path)

        self.n_ref = len(self.args.train_list)

        # 求 rays_o, rays_d
        images = ray_dataset.images
        poses = ray_dataset.poses
        intrinsics = ray_dataset.intrinsics
        focal = ray_dataset.focal
        H, W = ray_dataset.images.shape[1:3]
        n_rays_o, n_rays_d = [], []
        n_key_points_list = []
        n_depths = []
        n_kpts_rgbs = []

        if self.augmentation:
            n_aug_rays_o, n_aug_rays_d, n_aug_depths = [], [], []
            n_aug_key_points_list = []
            n_aug_point_cloud = []

        for idx in range(self.n_ref):
            image = images[idx]
            pose = poses[idx]
            K = intrinsics[idx]
            key_points, depth = self.projected2d(ori_point_cloud, K, pose)
            rays_o, rays_d = self.sample_rays_np(key_points, H, W, focal, pose)
            n_rays_o.append(rays_o)  # [N_key_points, 2]
            n_rays_d.append(rays_d)
            n_key_points_list.append(key_points)
            n_depths.append(depth)

            kpts_rgb = self.grid_sample_numpy(image, key_points)
            n_kpts_rgbs.append(kpts_rgb)

            if self.augmentation:
                aug_rays_o, aug_rays_d, aug_depths, aug_key_points, aug_point_cloud = self.augmentation_pts(key_points, ori_point_cloud, H, W, focal, pose)
                n_aug_rays_o.append(aug_rays_o)
                n_aug_rays_d.append(aug_rays_d)
                n_aug_depths.append(aug_depths)
                n_aug_key_points_list.append(aug_key_points)
                n_aug_point_cloud.append(aug_point_cloud)

        n_rays_o = np.concatenate(n_rays_o, axis=0)  # [n_samples, 3]
        n_rays_d = np.concatenate(n_rays_d, axis=0)  # [n_samples, 3]
        n_depths = np.concatenate(n_depths, axis=0)  # [n_samples, 1]


        n_depth_range_ratio = self.get_depth_range(ori_point_rgb, n_kpts_rgbs)
        n_depth_range_ratio = np.concatenate(n_depth_range_ratio, axis=0)
        n_depth_range_ratio = n_depth_range_ratio[:, np.newaxis]

        # ratio_max  = np.max(n_depth_range_ratio)
        # ratio_min = np.min(n_depth_range_ratio)
        # ratio_mean = np.mean(n_depth_range_ratio)


        if self.augmentation:
            n_aug_point_cloud = np.concatenate(n_aug_point_cloud, axis=0)
            n_aug_rays_o = np.concatenate(n_aug_rays_o, axis=0)
            n_aug_rays_d = np.concatenate(n_aug_rays_d, axis=0)
            n_aug_depths = np.concatenate(n_aug_depths, axis=0)

            if self.select_single_view_pts:
                ret = self.keep_single_view_pts(
                    n_aug_point_cloud,
                    poses,
                    intrinsics,
                    H, W,
                    focal
                )
                n_aug_rays_o, n_aug_rays_d, n_aug_depths, n_aug_key_points_list, n_aug_point_cloud = ret


        # 可视化特征点

        is_visulize = False
        if is_visulize:
            self.vis_per_kpts(n_key_points_list, n_kpts_rgbs, ori_point_rgb)
            if self.augmentation:

                self.vis_aug_key_points(n_key_points_list, n_aug_key_points_list)
                self.vis_aug_3d_pts(ori_point_cloud, n_aug_point_cloud)
            else:
                self.vis_key_points(n_key_points_list)


        key_rays_depth = np.concatenate((n_rays_o, n_rays_d, n_depths, n_depth_range_ratio), axis=-1).reshape((-1, 8))

        if self.augmentation:
            aug_key_rays_depth = np.concatenate((n_aug_rays_o, n_aug_rays_d, n_aug_depths), axis=-1).reshape((-1, 7))
            key_rays_depth = np.concatenate((key_rays_depth, aug_key_rays_depth), axis=0)


        self.key_points_rays = torch.tensor(key_rays_depth, device=device)


    def __len__(self):
        return self.key_points_rays.shape[0]

    def __getitem__(self, item):
        return self.key_points_rays[item]


    def augmentation_pts(self, key_points, ori_point_cloud, H, W, focal, pose):
        '''
        点云增强，生成新的点云做约束
        '''
        block_size = 64
        block_H = H // block_size
        block_W = W // block_size
        kpts_threshold_low = 30
        kpts_threshold_high = 60

        keys_block_ids = key_points[:, 1].astype(np.int32) // block_size * block_W + key_points[:, 0].astype(
            np.int32) // block_size
        block_kpts_nums = np.bincount(keys_block_ids, minlength=block_H * block_W)
        block_kpts_nums_mean = np.mean(block_kpts_nums).astype(np.int32)
        block_kpts_nums = block_kpts_nums.reshape((block_H, block_W))

        is_above_y, is_above_x = np.where(block_kpts_nums > kpts_threshold_high)

        aug_rays_o, aug_rays_d, aug_depths = [], [], []
        aug_kpts = []
        aug_point_cloud = []
        for i in range(0, block_H):
            for j in range(0, block_W):
                # 只计算边缘的点
                i_mirror = block_H - 1 - i if i > block_H // 2 else i
                j_mirror = block_W - 1 - j if j > block_W // 2 else j
                if (block_kpts_nums[i, j] < kpts_threshold_low) and (
                        i_mirror == 0 or j_mirror == 0 or i_mirror + j_mirror <= 2):
                    # find nearest block with enough key points
                    # Travelsal
                    dist = np.abs(is_above_y - i) + np.abs(is_above_x - j)
                    idx = np.argmin(dist)
                    i_near, j_near = is_above_y[idx], is_above_x[idx]
                    nearest_block_id = i_near * block_W + j_near
                    pts_ids = np.where(keys_block_ids == nearest_block_id)[0]

                    block_3d_pts = ori_point_cloud[pts_ids]
                    # z_mean_3d_pts = np.mean(block_3d_pts[:, 2])
                    # 计算待填充的block仿射变换的坐标边界
                    # dirs = [[0,0], [0,1], [1,0], [1,1]]

                    # 在 待填充的 block 中采样
                    kpts = np.array([j * block_size, i * block_size]) + np.random.rand(block_kpts_nums_mean,
                                                                                           2) * block_size

                    kpts = kpts.astype(np.float32)
                    # 计算变换
                    rays_o, rays_d = self.sample_rays_np(kpts, H, W, focal, pose)

                    new_pts_3d_z = np.random.choice(block_3d_pts[:, 2], block_kpts_nums_mean)
                    depths = (new_pts_3d_z - rays_o[:, 2]) / rays_d[:, 2]
                    depths = depths[:, np.newaxis]
                    new_pts_3d = rays_o + rays_d * depths

                    aug_rays_o.append(rays_o)
                    aug_rays_d.append(rays_d)
                    aug_depths.append(depths)

                    aug_kpts.append(kpts)

                    aug_point_cloud.append(new_pts_3d)


        aug_rays_o = np.concatenate(aug_rays_o, axis=0)
        aug_rays_d = np.concatenate(aug_rays_d, axis=0)
        aug_depths = np.concatenate(aug_depths, axis=0)

        aug_kpts = np.concatenate(aug_kpts, axis=0)

        aug_point_cloud = np.concatenate(aug_point_cloud, axis=0)


        return aug_rays_o, aug_rays_d, aug_depths, aug_kpts, aug_point_cloud

    def projected2d(self, points, K, pose, check_valid=False):
        '''
        # 将3d点投影到2d
        '''
        points_world = points[..., np.newaxis]  # [N, 3, 1]
        # 转换到相机坐标系
        # pose  opengl->opencv
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)
        pose = pose @ blender2opencv

        R_T = pose[:3, :3].T[np.newaxis, ...]  # [1, 3, 3]
        T = pose[:3, 3:][np.newaxis, ...]  # [1, 3, 1]
        points_camera = np.matmul(R_T, (points_world - T)) # [N, 3, 1]

        depth = points_camera[..., 2,:] # [N, 1]
        # 转换到像素坐标系
        K = K[np.newaxis, ...]  # [1, 3, 3]
        points_uv = np.matmul(K, points_camera)  # [N, 3, 1]

        # 归一化
        points_uv = points_uv.squeeze()
        points_uv = points_uv[...,:2] / points_uv[..., 2:]

        H, W = self.ray_dataset.images.shape[1:3]

        # 判断是否在图像内
        if check_valid:
            valid_mask_x = np.logical_and(points_uv[..., 0] > 0, points_uv[..., 0] < W)
            valid_mask_y = np.logical_and(points_uv[..., 1] > 0, points_uv[..., 1] < H)
            valid_mask = np.logical_and(valid_mask_x, valid_mask_y)
            valid_mask = valid_mask.squeeze()

            points_uv = points_uv[valid_mask]
            depth = depth[valid_mask]

        depth =np.abs(depth)

        return points_uv, depth

    def load_spc(self, spc_file_path):
        pos_list = list()

        if not os.path.exists(spc_file_path):
            raise ValueError(f"Files {spc_file_path} don't exist!")

        with open(spc_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if not line.startswith('#'):
                    line_nums = np.fromstring(line, sep=' ', dtype=np.float32)
                    pos = line_nums[1:4]

                    pos_list.append(pos)

            pos_npy = np.array(pos_list)

        return pos_npy

    def load_ply(self, ply_path):
        plydata = PlyData.read(ply_path)
        vertex = plydata['vertex']
        point_cloud = np.vstack((vertex['x'], vertex['y'], vertex['z'])).T
        point_rgb = np.vstack(
            (vertex['red'], vertex['green'], vertex['blue']),
            dtype=np.float32).T / 255.0
        return point_cloud, point_rgb

    def generate_spc(self, sample_num = 600):
        point_cloud_path = os.path.join(self.file_path, 'points3D.txt')
        point_cloud = self.load_spc(point_cloud_path)

        # 求z方向均值和方差
        z_mean = np.mean(point_cloud[:, 2])
        z_std = np.std(point_cloud[:, 2])

        x_mean = np.mean(point_cloud[:, 0])
        x_std = np.std(point_cloud[:, 0])

        y_mean = np.mean(point_cloud[:, 1])
        y_std = np.std(point_cloud[:, 1])

        x_bound = np.max(np.abs(point_cloud[:, 0] - x_mean))
        y_bound = np.max(np.abs(point_cloud[:, 1] - y_mean))

        samples = np.random.exponential(4.0, sample_num)
        normal_samples = 1.5 - 0.8 * samples / (np.max(samples) + np.finfo(float).eps)

        # 删去不符合z取值的点
        z_idx = np.logical_and(point_cloud[:, 2] > z_mean - 2 * z_std, point_cloud[:, 2] < z_mean + 2 * z_std)
        point_cloud_ori = point_cloud[z_idx]

        sample_idx = np.random.randint(0, point_cloud_ori.shape[0], sample_num)
        point_cloud_sampled = point_cloud_ori[sample_idx]

        x_max_ratio = x_bound / np.abs(point_cloud_sampled[:, 0] - x_mean)
        y_max_ratio = y_bound / np.abs(point_cloud_sampled[:, 1] - y_mean)

        max_ratio = np.min([x_max_ratio, y_max_ratio], axis=0)

        aug_point_cloud = np.zeros_like(point_cloud_sampled)
        dx = normal_samples * (point_cloud_sampled[:, 0] - x_mean) * max_ratio
        dy = normal_samples * (point_cloud_sampled[:, 1] - y_mean) * max_ratio
        radius = np.sqrt(dx * dx + dy * dy)
        angles = np.random.rand(sample_num) * 2 * np.pi
        aug_point_cloud[:, 0] = x_mean + radius * np.cos(angles)
        aug_point_cloud[:, 1] = y_mean + radius * np.sin(angles)
        aug_point_cloud[:, 2] = point_cloud_sampled[:, 2]

        aug_path = os.path.join(self.file_path, 'aug_points3D.npy')

        np.save(aug_path, aug_point_cloud)

        return aug_point_cloud


    def sample_rays_np(self, key_points, H, W, f, c2w):
        # i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        i, j = key_points[:, 0], key_points[:, 1]
        dirs = np.stack([(i - W * .5 + 0.5) / f, -(j - H * .5 + 0.5) / f, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    def vis_key_points(self, n_key_points_list):
        images = self.ray_dataset.images


        for idx in range(self.n_ref):
            result_img = images[idx]
            result_img = (result_img * 255).astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            key_point = n_key_points_list[idx].astype(np.int32)
            for i in range(key_point.shape[0]):
                cv2.circle(result_img, tuple(key_point[i]), 3, (0, 0, 255), 1)


            # result_img = cv2.resize(result_img, dsize=None, fy=0.5, fx=0.5)
            cv2.imshow(f'img_{idx}', result_img)
            cv2.waitKey(0)

    def vis_aug_key_points(self, n_key_points_list, aug_key_points_list):
        images = self.ray_dataset.images


        for idx in range(self.n_ref):
            result_img = images[idx]
            result_img = (result_img * 255).astype(np.uint8)
            result_img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            key_point = n_key_points_list[idx].astype(np.int32)
            for i in range(key_point.shape[0]):
                cv2.circle(result_img, tuple(key_point[i]), 3, (0, 0, 255), 1)
            aug_key_point = aug_key_points_list[idx].astype(np.int32)
            for i in range(aug_key_point.shape[0]):
                cv2.circle(result_img, tuple(aug_key_point[i]), 3, (255, 0, 0), 1)


            # result_img = cv2.resize(result_img, dsize=None, fy=0.5, fx=0.5)
            cv2.imshow(f'img_{idx}', result_img)
            cv2.waitKey(0)

    def vis_aug_3d_pts(self, point_cloud, aug_point_cloud):
        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)
        ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], c='r')

        ax.scatter(aug_point_cloud[:, 0], aug_point_cloud[:, 1], aug_point_cloud[:, 2], c='b')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    def keep_single_view_pts(self, aug_point_cloud, poses, intrinsics, H, W, focal):
        '''
        保留单视角点云
        '''
        n_rays_o = []
        n_rays_d = []
        n_key_points_list = []
        n_depths = []

        appear_in_view_nums = np.zeros(aug_point_cloud.shape[0])
        for idx in range(self.n_ref):
            pose = poses[idx]
            K = intrinsics[idx]
            key_points, depth = self.projected2d(aug_point_cloud, K, pose, check_valid=False)

            x_in_bound = np.logical_and(key_points[:, 0] >= 0, key_points[:, 0] < W)
            y_in_bound = np.logical_and(key_points[:, 1] >= 0, key_points[:, 1] < H)
            xy_in_bound = np.logical_and(x_in_bound, y_in_bound)
            appear_in_view_idx = np.where(xy_in_bound == True)[0]
            appear_in_view_nums[appear_in_view_idx] += 1

        single_view_idx = np.where(appear_in_view_nums == 1)[0]

        aug_point_cloud = aug_point_cloud[single_view_idx]
        for idx in range(self.n_ref):
            pose = poses[idx]
            K = intrinsics[idx]
            key_points, depth = self.projected2d(aug_point_cloud, K, pose)

            rays_o, rays_d = self.sample_rays_np(key_points, H, W, focal, pose)
            n_rays_o.append(rays_o)  # [N_key_points, 2]
            n_rays_d.append(rays_d)
            n_key_points_list.append(key_points)
            n_depths.append(depth)

        n_rays_o = np.concatenate(n_rays_o, axis=0)  # [n_samples, 3]
        n_rays_d = np.concatenate(n_rays_d, axis=0)
        n_depths = np.concatenate(n_depths, axis=0)


        return n_rays_o, n_rays_d, n_depths, n_key_points_list, aug_point_cloud

    def vis_per_kpts(self, n_key_points_list, n_kpts_rgb, point_cloud_rgb):
        images = self.ray_dataset.images
        for idx in range(n_key_points_list[0].shape[0]):
            result_list = []
            for i in range(self.n_ref):
                result_img = images[i]
                result_img = (result_img * 255).astype(np.uint8)
                # rgb_temp = self.get_pixel_bilinear(result_img, n_key_points_list[i][idx])
                coords = tuple(n_key_points_list[i][idx].astype(np.int32))
                # rgb_temp_1 = result_img[coords[1], coords[0]]
                # rgb_list.append(rgb_temp)
                cv2.circle(result_img, coords, 3, (255, 0, 0), 1)
                result_list.append(result_img)

            rgb_array = np.array(
                [
                    n_kpts_rgb[0][idx],
                    n_kpts_rgb[1][idx],
                    n_kpts_rgb[2][idx]
                ]
            )
            pts_rgb = point_cloud_rgb[idx].astype(np.float32)
            err_1 = []
            err_1.append(np.mean(np.abs(rgb_array[0,:]-pts_rgb)))
            err_1.append(np.mean(np.abs(rgb_array[1,:]-pts_rgb)))
            err_1.append(np.mean(np.abs(rgb_array[2,:]-pts_rgb)))


            err_2 = np.mean(np.std(rgb_array, axis=0))

            if err_2 > 0.0:

                print(f"{idx}'s RGB is ")
                print("point is ", point_cloud_rgb[idx])
                print("Ref 1 is ", rgb_array[0,:], "Error is ", err_1[0])
                print("Ref 2 is ", rgb_array[1,:], "Error is ", err_1[1])
                print("Ref 3 is ", rgb_array[2,:], "Error is ", err_1[2])
                print("Var is ", err_2)



                all_result_img = cv2.hconcat(result_list)
                # all_result_img = cv2.resize(all_result_img, dsize=None, fy=0.5, fx=0.5)
                all_result_img = cv2.cvtColor(all_result_img, cv2.COLOR_RGB2BGR)
                cv2.imshow(f'kpts_{idx}', all_result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


    def get_pixel_bilinear(self, image, coords):
        '''
        images: [H, W, 3], rgb
        coords: [x, y]
        '''

        x, y = coords
        # 获取图像的宽度和高度
        height, width, channels = image.shape

        # 找到最近的四个整数坐标
        x1 = int(np.floor(x))
        x2 = int(np.ceil(x))
        y1 = int(np.floor(y))
        y2 = int(np.ceil(y))

        # 确保坐标在图像范围内
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))

        # 获取四个点的颜色值
        Q11 = image[y1, x1]
        Q12 = image[y2, x1]
        Q21 = image[y1, x2]
        Q22 = image[y2, x2]

        # 计算权重
        weight_x = x - x1
        weight_y = y - y1

        # 计算双线性插值
        color = (1 - weight_x) * (1 - weight_y) * Q11 + \
                (1 - weight_x) * weight_y * Q12 + \
                weight_x * (1 - weight_y) * Q21 + \
                weight_x * weight_y * Q22

        return color

    def grid_sample_numpy(self, image, coords):
        '''
        images: [H, W, 3], rgb
        coords: [N, 2]
        '''

        x, y = coords[:, 0], coords[:, 1]
        # 获取图像的宽度和高度
        height, width, channels = image.shape

        # 找到最近的四个整数坐标
        x1 = np.floor(x)
        x2 = np.ceil(x)
        y1 = np.floor(y)
        y2 = np.ceil(y)


        # 确保坐标在图像范围内
        x1 = np.clip(x1, 0, width - 1).astype(np.int32)
        x2 = np.clip(x2, 0, width - 1).astype(np.int32)
        y1 = np.clip(y1, 0, height - 1).astype(np.int32)
        y2 = np.clip(y2, 0, height - 1).astype(np.int32)


        # 获取四个点的颜色值
        Q11 = image[y1, x1]
        Q12 = image[y2, x1]
        Q21 = image[y1, x2]
        Q22 = image[y2, x2]

        # 计算权重
        weight_x = x - x1
        weight_y = y - y1

        weight_x = weight_x[:, np.newaxis].astype(np.float32)
        weight_y = weight_y[:, np.newaxis].astype(np.float32)

        # 计算双线性插值
        color = (1 - weight_x) * (1 - weight_y) * Q11 + \
                (1 - weight_x) * weight_y * Q12 + \
                weight_x * (1 - weight_y) * Q21 + \
                weight_x * weight_y * Q22

        return color

    def get_depth_range(self, ori_point_rgb, n_kpts_rgbs):
        '''
        ori_point_rgb: [N, 3]
        n_kpts_rgb: [3, N, 3]
        err_1 in (0, 1)
        '''
        rgb_array = np.array(n_kpts_rgbs)

        err_2 = np.mean(np.std(rgb_array, axis=0), axis=1)

        # err_temp = np.mean(np.std(rgb_array[:,0,:].squeeze(), axis=0))

        # err_2_max = np.max(err_2)
        # err_2_min = np.min(err_2)

        n_depth_range_ratio = []

        for idx in range(self.n_ref):
            err_1 = np.mean(np.abs(n_kpts_rgbs[idx] - ori_point_rgb), axis=1)
            n_depth_range_ratio.append(err_1+err_2)

        return n_depth_range_ratio

import config
if __name__ == '__main__':
    args = config.load_config()
    train_dataset = LevirNVSDataset(args, 'train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # spc_dataset = LevirSPCDataset(args, device)
    # spc_dataset.vis_key_points(train_dataset)
    dataset = LevirSPCAugDataset(args, train_dataset, device)
    print(len(dataset))