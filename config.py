import argparse
import gin
from dataclasses import dataclass, field
from typing import List, Literal


@gin.configurable()
@dataclass
class Config:
    gin_configs: List[str] = field(default_factory=lambda: [])  # 配置文件路径 列表
    rootdir: str = r'./exp'
    ckpt_dir: str = r'./checkpoint'
    expname: str = 'Levir_test'
    distributed: bool = False
    initial_ckpt: bool = False

    torch_seed: int = 666
    np_seed: int = 999
    torch_cuda_seed: int = 666

    data_path: str = r"./data"  # dataset path
    dataset: str = "levir_nvs"  # dataset_name
    scene_id: str = "scene_000"  # scene_id
    train_list: List[int] = field(default_factory=lambda: [0, 7, 14])  # training_idx
    val_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])  # validation idx
    img_center_crop: bool = False  # crop image to save GPU Memory
    img_resize: bool = False  # resize image to save GPU Memory

    batch_size: int = 1024  # batchsize
    key_pts_batch_size: int = 64  # keypoint batchsize
    chunk_size: int = 512 * 8  # chunksize for rendering image, x times of image width
    sampler_type: Literal[0, 1, 2] = field(default=0)  # sample type 0 for const, 1 for anneal, 2 for grid

    keypoint_weight: float = 0.001  # whether to use keypoints to constrain
    keypoint_iters: int = 10000  # keypoint iters
    key_pts_augmentation: bool = False  # whether to use keypoint augmentation
    kpts_range: float = 0.0  # keypoint range
    kpts_depth_max_ratio: float = 0.4  # keypoint depth max ratio
    kpts_weight_type: str = 'none'  # 'L1' 'L2'  'none'

    select_single_view_pts: bool = False  # whether to select single view keypoint

    bounds: List[float] = field(default_factory=lambda: [90., 140.])  # t_near and t_far
    init_bounds: List[float] = field(default_factory=lambda: [0., 150.])  # inintial t_near and t_far

    anneal_nearfar_steps: int = 2000  # when to stop annealing
    anneal_nearfar_perc: float = 0.2  # percentage for near/far annealing at start.
    anneal_mid_perc: float = 0.5  # Perc for near/far mid point.

    occ_grid_resolution: int = 128  # grid resolution
    occ_aabb: int = 50  # occ_aabb = (H/2)/focal*camera_height
    occ_warmup: bool = False  # whether to warm up without the occ
    occ_warmup_steps: int = 1000  # occ delay steps

    shrink_scale: int = 1  # scale the bounds

    no_reload: bool = False  # do not load checkpoints
    no_load_opt: bool = False  # do not load optimizer when reloading
    no_load_scheduler: bool = False  # do not load scheduler when reloading
    ckpt_path: str = None  # specific ckpt path,eg xxx.pth

    render_n_samples: int = 128  # samples num

    lr: float = 1e-4  # learning rate
    triplane_lr_scale: float = 1.0  # triplane lr scale
    lr_decay_step: int = 50000  # learning rate decay step
    lr_decay_gamma: float = 0.5  # learning rate decay gamma

    finetune_encoder: bool = False  # whether finetune encoder
    lr_encoder: float = 5e-4  # image encoder learning rate

    n_iters: int = 50000  # training epochs

    i_print: int = 100  # frequency of terminal printout
    i_img: int = 10000  # frequency of tensorboard image logging
    i_weights: int = 10000  # frequency of weight ckpt saving

    use_smoothness: bool = False  # whether to use depth smooth
    smoothness_iters: int = 10000  # smoothness iters

    depth_smooth_weight: float = 0.001  # depth loss weight

    vis_heatmap: bool = False  # visualize heatmap

    visualize_density: bool = False

    multi_step_lr: bool = False


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=False)

    if args.ginc:
        gin.bind_parameter('Config.gin_configs', args.ginc)
    gin.finalize()

    config = Config()

    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ginc",
        action="append",
        help="gin config file",
    )
    parser.add_argument(
        "--ginb",
        action="append",
        help="gin bindings",
    )
    args = parser.parse_args()

    ginbs = []
    if args.ginb:
        ginbs.extend(args.ginb)
    gin.parse_config_files_and_bindings(args.ginc, ginbs, finalize_config=False)

    if args.ginc:
        gin.bind_parameter('Config.gin_configs', args.ginc)
    gin.finalize()

    config = Config()
    if config.gin_configs:
        print("yes")
    conf = gin.operative_config_str()
    with open(r'configs/config.gin', 'w') as f:
        f.write(conf)
