import gin
import numpy as np
import torch

from einops import rearrange
from internal.render_ray import render_rays

from kornia.losses import inverse_depth_smoothness_loss

@gin.configurable()
class Smoother:
    def __init__(
        self,
        args,
        test_dataset,
        ref_dataset,
        depth_type: str = 'depth', # depth or inverse_depth
        loss_type: str = 'inverse_depth_smoothness', # inverse_depth_smoothness or depth_smoothness
        patch_size: int = 16,
        patch_num: int = 1,
        down_scale: int = 4,
        render_type: str = 'train', # 'train' or 'val' or 'train_val'
    ):
        self.args = args

        self.depth_type = depth_type
        self.render_type = render_type
        self.patch_size = patch_size
        self.patch_num = patch_num
        self.down_scale = down_scale
        self.loss_type = loss_type


        self.ref_dataset = ref_dataset
        self.test_dataset = test_dataset

    def __call__(self, model, train_sample_kwargs):
        rays_patch = self.sample_rays()
        rays_o, rays_d, rgb_gt = torch.chunk(rays_patch, 3, dim=-1)
        rays_od = (rays_o.reshape(-1,3), rays_d.reshape(-1,3))



        ret = render_rays(
            model.net, rays_od, self.ref_dataset,
            train_sample_kwargs
        )

        # idepth = 1./ (ret['depth']+1e-6)

        rgb_map = rgb_gt.unsqueeze(0).permute(0, 3, 1, 2) #[B,C,H,W]
        if self.depth_type == 'depth':
            depth_map = ret['depth']
        elif self.depth_type == 'inverse_depth':
            depth_map = 1./(ret['depth']+0.01)
        else:
            raise ValueError('Unknown depth type')

        depth_map = rearrange(
            depth_map, '(b p q) c ->b c p q', c=1, p=self.patch_size, q=self.patch_size)  # [B,N,H,W]

        if self.loss_type == 'inverse_depth_smoothness':
            loss_smooth = inverse_depth_smoothness_loss(depth_map, rgb_map)
        else:
            raise ValueError('Unknown loss type')
        return loss_smooth


    def sample_rays(self):
        # select a random view
        if self.render_type == 'train':
            idx = np.random.choice(self.args.train_list)
        elif self.render_type == 'val':
            val_list = [i for i in range(20) if i not in self.args.train_list]
            idx = np.random.choice(val_list)
        elif self.render_type == 'train_val':
            idx = np.random.choice(20)
        else:
            raise ValueError('Unknown render type')

        rays_full = self.test_dataset.rays[idx]  # [H,W,9]
        h,w = rays_full.shape[:2]
        ll = np.random.randint(
            0, w - (self.patch_size - 1) * self.down_scale - 1)
        up = np.random.randint(
            0, h - (self.patch_size - 1) * self.down_scale - 1)

        rays_patch = rays_full[ll:ll + (self.patch_size - 1) * self.down_scale + 1:self.down_scale, up:up + (
                self.patch_size - 1) * self.down_scale + 1:self.down_scale, :]


        return rays_patch # [H,W,9]


