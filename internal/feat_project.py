import torch
import torch.nn.functional as F
import gin


@gin.configurable()
class Projector():
    def __init__(
            self,
            train_dataset,
            ref_feature,
            device,
            padding_mode="zeros"
    ):
        images = train_dataset.images
        c2w = train_dataset.poses
        f = train_dataset.focal

        img_size, _ = images.shape[1:3]

        train_images = torch.tensor(images, device=device, dtype=torch.float32).permute(0, 3, 1, 2)

        self.train_images = train_images

        # self.reference = reference
        self.scale = (img_size / 2) / f
        self.n = c2w.shape[0]
        self.R_t = torch.tensor(c2w[:, :3, :3], device=device).permute(0, 2, 1)  # [3,3,3]
        self.camera_pos = torch.tensor(c2w[:, :3, -1], device=device)
        self.c2w = c2w
        self.img_size = img_size
        self.f = f
        self.ref_feature = ref_feature
        self.n_refs = self.ref_feature.shape[0]

        self.neareat_idxs = None
        self.padding_mode = padding_mode

    def feature_matching(self, pos):
        n_rays, n_samples, _ = pos.shape
        pos = pos.unsqueeze(dim=0).expand([self.n, n_rays, n_samples, 3])  # [3,nr,ns,3]
        camera_pos = self.camera_pos[:, None, None, :]  # [3,1,1,3]
        camera_pos = camera_pos.expand_as(pos)  # [3,nr,ns,3]
        ref_pos = torch.einsum("kij,kbsj->kbsi", self.R_t, pos - camera_pos)  # [3, nr,ns, 3]
        uv_pos = ref_pos[..., :-1] / -ref_pos[..., -1:] / self.scale
        uv_pos[..., 1] *= -1.0

        if self.n_refs <= 3:
            nearest_ref_feature = self.ref_feature
        else:
            nearest_ref_feature = self.ref_feature[self.neareat_idxs]

        return F.grid_sample(nearest_ref_feature, uv_pos, align_corners=True, padding_mode=self.padding_mode)

    def update_ref_feature(self, ref_feature):
        self.ref_feature = ref_feature

    def update_nearest_idx(self, current_idx):

        neareat_idxs = [(current_idx - 1) % self.n_refs, current_idx, (current_idx + 1) % self.n_refs]
        self.neareat_idxs = neareat_idxs
