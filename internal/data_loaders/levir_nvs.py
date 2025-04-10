import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class LevirNVSDataset(Dataset):
    def __init__(self, args, mode):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.mode = mode  # "train" or "validation" or "test"
        self.scale = 1  # scale factor, resize to 1/scale
        self.scene_id = args.scene_id
        self.selected_views = {
            'train': args.train_list,
            'validation': args.val_list,
            'test': list(range(20)),
            'render_track': list(range(3))
        }
        # self.val_list = [i for i in range(20) if i not in self.train_list]

        self.dataset_filepath = args.data_path + r'/LEVIR_NVS'
        self.neighbor_view_ids = ["001", "002", "003", "004", "005", "006", "007", "008", "009", "010", "011", "012",
                                  "013", "014", "015", "016", "017", "018", "019", "020"]

        images_tgt, Ks_tgt, Es_tgt = [], [], []
        opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

        for i, neighbor_view_id in enumerate(self.neighbor_view_ids):
            if i in self.selected_views[self.mode]:
                image = self.read_image(
                    os.path.join(self.dataset_filepath, self.scene_id, "Images/{}.png".format(neighbor_view_id))
                )
                K_tgt, E_tgt, depth_min_tgt, depth_max_tgt = self.read_camera_parameters(
                    os.path.join(self.dataset_filepath, self.scene_id, "Cams/{}.txt".format(neighbor_view_id))
                )

                E_tgt = np.linalg.inv(E_tgt)  # w2c--> c2w
                E_tgt = E_tgt @ opencv2blender  # c2w下的  opencv -->  opengl

                images_tgt.append(image)
                Ks_tgt.append(K_tgt)
                Es_tgt.append(E_tgt)

        self.images = np.stack(images_tgt)
        Ks_tgt = np.stack(Ks_tgt)
        self.poses = np.stack(Es_tgt)
        self.focal = (Ks_tgt[0][0][0] / self.scale).astype(np.float32)
        self.intrinsics = Ks_tgt

        if self.mode == 'render_track':
            self.poses = self.read_rendered_pathes()

        self.rays = self.create_ray_batches()

    def __len__(self):
        return self.rays.shape[0]

    def __getitem__(self, item):
        return self.rays[item]

    def read_image(self, filename):
        image = Image.open(filename)
        W, H = image.size

        if self.args.img_center_crop:
            left = W // 4
            top = H // 4
            right = 3 * W // 4
            bottom = 3 * H // 4
            image = image.crop((left, top, right, bottom))
            W, H = image.size

        if self.args.img_resize:
            image = image.resize((W // 2, H // 2))
            self.scale = 2

        # normalize 0-255 to 0-1
        image = np.array(image, dtype=np.float32) / 255.
        # print(image.shape)
        return image

    def read_camera_parameters(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        extrinsic = np.fromstring(",".join(lines[1:5]), dtype=np.float32, sep=",").reshape(4, 4)
        intrinsic = np.fromstring(",".join(lines[7:10]), dtype=np.float32, sep=",").reshape(3, 3)
        depth_min, depth_max = [float(item) for item in lines[11].split(",")]
        return intrinsic, extrinsic, depth_min, depth_max

    def create_ray_batches(self):
        # train 按 batch 分配
        if self.mode == 'train':
            H, W, = self.images.shape[1:3]
            # print("Create Ray batches!")
            rays_o_list = list()
            rays_d_list = list()
            rays_rgb_list = list()
            for i in range(self.images.shape[0]):
                img = self.images[i]
                pose = self.poses[i]
                rays_o, rays_d = self.sample_rays_np(H, W, self.focal, pose)
                rays_o_list.append(rays_o.reshape(-1, 3))
                rays_d_list.append(rays_d.reshape(-1, 3))
                rays_rgb_list.append(img.reshape(-1, 3))
            rays_o_npy = np.concatenate(rays_o_list, axis=0)
            rays_d_npy = np.concatenate(rays_d_list, axis=0)
            rays_rgb_npy = np.concatenate(rays_rgb_list, axis=0)
            rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=1), device=self.device)
            # [N*H*W,9]
        elif self.mode == 'validation' or self.mode == 'test':
            # with selected imgs 保持图像完整
            H, W, = self.images.shape[1:3]
            # print("Create Ray batches!")
            rays_o_list = list()
            rays_d_list = list()
            rays_rgb_list = list()
            for i in range(self.images.shape[0]):
                img = self.images[i]
                pose = self.poses[i]
                rays_o, rays_d = self.sample_rays_np(H, W, self.focal, pose)
                rays_o_list.append(rays_o)
                rays_d_list.append(rays_d)
                rays_rgb_list.append(img)
            rays_o_npy = np.stack(rays_o_list, axis=0)
            rays_d_npy = np.stack(rays_d_list, axis=0)
            rays_rgb_npy = np.stack(rays_rgb_list, axis=0)
            rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy, rays_rgb_npy], axis=-1), device=self.device)
            # [20,H,W,9]

        elif self.mode == 'render_track':
            # with selected imgs 保持图像完整
            H, W, = self.images.shape[1:3]
            # print("Create Ray batches!")
            rays_o_list = list()
            rays_d_list = list()
            for i in range(self.poses.shape[0]):
                pose = self.poses[i]
                rays_o, rays_d = self.sample_rays_np(H, W, self.focal, pose)
                rays_o_list.append(rays_o)
                rays_d_list.append(rays_d)

            rays_o_npy = np.stack(rays_o_list, axis=0)
            rays_d_npy = np.stack(rays_d_list, axis=0)
            rays = torch.tensor(np.concatenate([rays_o_npy, rays_d_npy], axis=-1), device=self.device)
            # [100,H,W,6]
        else:
            raise ValueError("Wrong mode for Levir-NVS Dataset!")
        return rays

    def sample_rays_np(self, H, W, f, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i - W * .5 + 0.5) / f, -(j - H * .5 + 0.5) / f, -np.ones_like(i)], -1)
        rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    def read_rendered_pathes(self):
        track_folder = os.path.join(self.dataset_filepath, self.scene_id, "Track")
        nums = len(os.listdir(track_folder))
        Es_tgt = []
        opencv2blender = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=np.float32)

        for i in range(nums):
            K_tgt, E_tgt, depth_min_tgt, depth_max_tgt = self.read_camera_parameters(
                os.path.join(track_folder, f'{i + 1:03d}.txt')
            )
            E_tgt = np.linalg.inv(E_tgt)  # w2c--> c2w
            E_tgt = E_tgt @ opencv2blender  # c2w下的  opencv -->  opengl

            Es_tgt.append(E_tgt)

        rendered_poses = np.stack(Es_tgt)

        return rendered_poses
