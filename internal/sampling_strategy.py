import torch
from enum import Enum, unique
from internal.ray_sampler import sampler_list
from utils import contraction

@unique
class SamplerType(Enum):
    CONST = 0
    ANNEAL = 1
    GRID = 2

class TrainSampler:
    def __init__(self, args, device, img_size):
        self.args = args
        self.device = device

        self.sampler = sampler_list[args.sampler_type](args, device, img_size)
        self.sample_params = None
        self.val_sample_params = None


    def __call__(self, global_step):

        if self.args.sampler_type == SamplerType.CONST.value:
            # 常数区间
            if self.sample_params is None:
                self.sample_params = self.sampler.get_sample_params(mode='train') #避免重复计算
        elif self.args.sampler_type == SamplerType.ANNEAL.value:
            # 退火方法
            if global_step <= self.args.anneal_nearfar_steps or self.sample_params is None:
                # anneal process
                self.sample_params = self.sampler.get_sample_params(global_step, mode='train')
        elif self.args.sampler_type == SamplerType.GRID.value:
            # occ方法
            # global_step 必须从0开始以进行初始化
            if self.args.occ_warmup is True and global_step < self.args.occ_warmup_steps:
                if self.sample_params is None:
                    self.sample_params = self.sampler.get_sample_params(mode='train')
            elif self.args.occ_warmup is True and global_step == self.args.occ_warmup_steps:
                self.sample_params = self.sampler.get_grid_params(global_step=1) # 初始化
            else:
                self.sample_params = self.sampler.get_grid_params(global_step)
        else:
            raise ValueError('Invalid sampler type!')

        return self.sample_params


    def set_occ_fn(self, model, reference_dataset):
        render_step_size = self.sampler.render_step_size
        # density_query_fn = lambda x: model.net.query_opacity(
        #     x, render_step_size, reference_dataset
        # )

        density_query_fn = lambda x: model.net.query_density(
            x=contraction(x, self.sampler.grid_kwargs),
            ref_feat=reference_dataset.feature_matching(x.unsqueeze(dim=0)),
        )['density']*render_step_size
        self.sampler.set_density_query_fn(density_query_fn)



class TestSampler:
    def __init__(self, args, device, img_size):
        self.args = args
        self.device = device

        self.sampler = sampler_list[0](args, device, img_size) # const
        self.sample_params = None

    def __call__(self):
        if self.sample_params is None:
            self.sample_params = self.sampler.get_sample_params(mode='val')
        return self.sample_params

def set_sampling_space(args, bounds, device, full_image=False, img_size=None, chunk_size=None):
    '''set bounds for each rays

    :param bounds: [t_near, t_far]
    :param batch_size:
    :param render_n_samples: number of points each ray
    :param device:
    :param full_image: for validation
    :param img_size:
    :return:
    '''

    t_bounds = torch.tensor(bounds, device=device, dtype=torch.float32)
    render_step_size = (t_bounds[1] - t_bounds[0]) / 64
    if not full_image:
        N_F = torch.broadcast_to(t_bounds, (args.batch_size, 2))
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


def anneal_nearfar(args, it):
    """Anneals near and far plane."""
    n_steps = args.anneal_nearfar_steps
    init_perc = args.anneal_nearfar_perc
    mid_perc = args.anneal_mid_perc

    near_final, far_final = args.bounds

    mid = near_final + mid_perc * (far_final - near_final)

    near_init = mid + init_perc * (near_final - mid)
    far_init = mid + init_perc * (far_final - mid)

    weight = min(it * 1.0 / n_steps, 1.0)

    near_i = near_init + weight * (near_final - near_init)
    far_i = far_init + weight * (far_final - far_init)
    return near_i, far_i


