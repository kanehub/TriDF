import torch
import math

from nerfacc import ContractionType, OccupancyGrid


class RaySampler:
    def __init__(self, args, device, img_size):
        self.args = args
        self.device = device
        self.img_size = img_size

    def get_sample_params(self, mode):
        if mode == 'train':
            N_F, render_step_size = self.set_sampling_space(self.args.bounds, self.device)
        elif mode == 'val':
            N_F, render_step_size = self.set_sampling_space(self.args.bounds, self.device, full_image=True,
                                                            img_size=self.img_size, chunk_size=self.args.chunk_size)

        sample_kwargs = {
            'mode': mode,
            'near_far': N_F,
            'render_step_size': render_step_size,
            'border': self.args.occ_aabb,
        }

        return sample_kwargs

    def set_sampling_space(self, bounds, device, full_image=False, img_size=None, chunk_size=None):
        '''set bounds for each ray
        :param bounds: [t_near, t_far]
        :param batch_size:
        :param render_n_samples: number of points each ray
        :param device:
        :param full_image: for validation
        :param img_size:
        :return:
        '''
        t_bounds = torch.tensor(bounds, device=device, dtype=torch.float32)
        render_step_size = (t_bounds[1] - t_bounds[0]) / self.args.render_n_samples
        if not full_image:
            N_F = torch.broadcast_to(t_bounds, (self.args.batch_size, 2))
        else:
            if img_size is not None:
                if chunk_size is not None:
                    # keep same with chunk size
                    N_F = torch.broadcast_to(t_bounds, (chunk_size, 2))
                else:
                    # keep same with img size
                    N_F = torch.broadcast_to(t_bounds, (img_size, 2))
            else:
                raise ValueError('Invalid img_size!')
        #  N_F (bs,2)
        return N_F, render_step_size


class AnnealSampler(RaySampler):
    def __init__(self, args, device, img_size):
        super(AnnealSampler, self).__init__(args, device, img_size)

    def get_sample_params(self, global_step, mode='train'):
        if mode == 'train':
            anneal_bounds = self.anneal_nearfar(global_step)
            N_F, render_step_size = self.set_sampling_space(anneal_bounds, self.device)
        elif mode == 'val':
            N_F, render_step_size = self.set_sampling_space(self.args.bounds, self.device, full_image=True,
                                                            img_size=self.img_size)

        sample_kwargs = {
            'mode': mode,
            'near_far': N_F,
            'render_step_size': render_step_size,
            'border': self.args.occ_aabb
        }

        return sample_kwargs

    def anneal_nearfar(self, it):
        """Anneals near and far plane."""
        n_steps = self.args.anneal_nearfar_steps
        init_perc = self.args.anneal_nearfar_perc
        mid_perc = self.args.anneal_mid_perc

        near_final, far_final = self.args.bounds

        mid = near_final + mid_perc * (far_final - near_final)

        near_init = mid + init_perc * (near_final - mid)
        far_init = mid + init_perc * (far_final - mid)

        weight = min(it * 1.0 / n_steps, 1.0)

        near_i = near_init + weight * (near_final - near_init)
        far_i = far_init + weight * (far_final - far_init)
        return near_i, far_i


class GridSampler(RaySampler):
    def __init__(self, args, device, img_size):
        super(GridSampler, self).__init__(args, device, img_size)
        render_n_samples = args.render_n_samples
        border = args.occ_aabb
        sign_list = [-1, -1, -1, 1, 1, 1]
        aabb = [float(border * item) for item in sign_list]
        contraction_type = ContractionType.AABB
        self.scene_aabb = torch.tensor(aabb, dtype=torch.float32, device=device)
        self.render_step_size = (
                (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
                * math.sqrt(3)
                / render_n_samples
        ).item()

        grid_resolution = args.occ_grid_resolution
        self.occupancy_grid = OccupancyGrid(
            roi_aabb=aabb,
            resolution=grid_resolution,
            contraction_type=contraction_type,
        ).to(device)

        self.density_query_fn = None

        self.grid_kwargs = {
            'occupancy_grid': self.occupancy_grid,
            'scene_aabb': self.scene_aabb,
            'render_step_size': self.render_step_size,
            'border': self.args.occ_aabb
        }

    def update_grid(self, step):
        if self.density_query_fn is not None:
            self.occupancy_grid.every_n_step(step=step, occ_eval_fn=self.density_query_fn)
        else:
            raise ValueError('density_query_fn is not defined!')

    def get_grid_params(self, global_step):
        self.update_grid(step=global_step-1)
        self.grid_kwargs['occupancy_grid'] = self.occupancy_grid

        return self.grid_kwargs

    def set_density_query_fn(self, density_query_fn):
        self.density_query_fn = density_query_fn


sampler_list = [RaySampler, AnnealSampler, GridSampler]
