import os
import time
from itertools import cycle
import numpy as np
import torch
from loguru import logger
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from evaluation.eval import eval_train
from internal.data_loaders import dataset_dict
from internal.feat_project import Projector
from internal.model import TriDFModel
from internal.render_ray import render_rays
from internal.sampling_strategy import SamplerType, TestSampler, TrainSampler
from internal.smoothness import Smoother
from utils import (img2mse, init_output_folder, mse2psnr,
                   save_gin_config)
from utils.visualization import log_view_to_tb


def init_logging(args):
    out_folder = os.path.join(args.rootdir, args.expname)
    logger_filepath = os.path.join(out_folder, 'train.log')
    # logging
    logger.remove(handler_id=None)  # do not show in console
    logger.add(
        sink=logger_filepath,
        level='DEBUG',
        encoding='utf-8',
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

    logger.info("====================== Training Infomation ====================")
    logger.info("spc init, use occ grid")
    logger.info("========================= Training Start =======================")


def train(args):
    # configs
    out_folder, ckpt_folder = init_output_folder(args)

    # logging
    init_logging(args)

    logger.info('outputs will be saved to {}'.format(out_folder))
    print('outputs will be saved to {}'.format(out_folder))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    torch.cuda.manual_seed(args.torch_cuda_seed)

    # create training dataset
    print("Process rays data for training!")
    logger.info("Process rays data for training!")
    print('training dataset: {}'.format(args.dataset))
    logger.info('training dataset: {}'.format(args.dataset))

    train_dataset = dataset_dict[args.dataset](args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)

    aug_spc_dataset = dataset_dict['levir_spc_aug'](
        args, train_dataset, device,
        is_augmentation=args.key_pts_augmentation,
        select_single_view_pts=args.select_single_view_pts,
        resample=False)
    aug_spc_dataloader = DataLoader(aug_spc_dataset, batch_size=args.key_pts_batch_size, drop_last=True, shuffle=True)
    aug_spc_dataloader_iterator = iter(cycle(aug_spc_dataloader))
    print("Length of aug_spc_dataset: ", len(aug_spc_dataset))

    val_dataset = dataset_dict[args.dataset](args, 'validation')
    val_loader = DataLoader(val_dataset, batch_size=1)
    val_loader_iterator = iter(cycle(val_loader))

    test_dataset = dataset_dict[args.dataset](args, 'test')

    # create model
    model = TriDFModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)

    train_images = train_dataset.images
    train_images = torch.tensor(train_images, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    # reference dataset
    with torch.no_grad():
        featmaps = model.feature_net(train_images)
    reference_dataset = Projector(train_dataset, featmaps, device)

    # load initailization ckpt
    if args.initial_ckpt:
        model.load_spc_ckpt()

    # Ray Sampler
    sampler = TrainSampler(args, device, reference_dataset.img_size)
    val_sampler = TestSampler(args, device, reference_dataset.img_size)
    val_sample_kwargs = val_sampler()

    if args.sampler_type == SamplerType.GRID.value:
        sampler.set_occ_fn(model, reference_dataset)

    # criterion
    mse = torch.nn.MSELoss()

    # logging file
    writer = SummaryWriter(out_folder)
    print('saving tensorboard files to {}'.format(out_folder))
    logger.info('saving tensorboard files to {}'.format(out_folder))
    scalars_to_log = {}

    # regulizar
    if args.use_smoothness:
        depth_regularizer = Smoother(args, test_dataset, reference_dataset)

    save_gin_config(out_folder)
    global_step = model.start_step + 1

    epoch = int(global_step // args.batch_size)

    train_start_time = time.time()
    while global_step < model.start_step + args.n_iters + 1:
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", ncols=85) as p_bar:
            for train_rays in train_loader:

                assert train_rays.shape == (args.batch_size, 9)
                rays_o, rays_d, target_rgb = torch.chunk(train_rays, 3, dim=-1)
                rays_od = (rays_o, rays_d)

                train_sample_kwargs = sampler(global_step)

                if args.finetune_encoder:
                    featmaps = model.feature_net(train_images)
                    reference_dataset.update_ref_feature(featmaps)
                    if args.sampler_type == SamplerType.GRID.value:
                        sampler.set_occ_fn(model, reference_dataset)

                ret = render_rays(
                    model.net, rays_od, reference_dataset,
                    train_sample_kwargs
                )

                loss_rgb = mse(ret['rgb'], target_rgb)
                scalars_to_log['loss/loss_rgb'] = loss_rgb.item()
                loss = loss_rgb

                if args.keypoint_weight > 0 and global_step < args.keypoint_iters:
                    keypoint_data = next(aug_spc_dataloader_iterator)
                    k_rays_o, k_rays_d, depths_ret = torch.chunk(keypoint_data, 3, dim=-1)
                    k_rays_od = (k_rays_o, k_rays_d)
                    k_ret = render_rays(
                        model.net, k_rays_od, reference_dataset,
                        train_sample_kwargs
                    )
                    tgt_depths, tgt_depth_ratios = torch.chunk(depths_ret, 2, dim=-1)

                    if args.kpts_weight_type == 'L1':
                        weight = torch.clamp(1 - tgt_depth_ratios, min=0, max=1)
                        sqrt_weight = torch.sqrt(weight)
                        loss_depth = mse(k_ret['depth'] * sqrt_weight, tgt_depths * sqrt_weight)
                    elif args.kpts_weight_type == 'L2':
                        weight = torch.clamp(1 - tgt_depth_ratios, min=0, max=1)
                        loss_depth = mse(k_ret['depth'] * weight, tgt_depths * weight)
                    else:
                        loss_depth = mse(k_ret['depth'], tgt_depths)

                    loss += args.keypoint_weight * loss_depth
                    scalars_to_log['loss/loss_keypoint'] = loss_depth.item()

                if args.use_smoothness and global_step > args.smoothness_iters:
                    loss_smooth = depth_regularizer(model, train_sample_kwargs)
                    loss += args.depth_smooth_weight * loss_smooth
                    scalars_to_log['loss/loss_smooth'] = loss_smooth.item()

                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                model.scheduler.step()

                scalars_to_log['lr'] = model.scheduler.get_last_lr()[-1]
                scalars_to_log['lr/triplane'] = model.scheduler.get_last_lr()[0]

                scalars_to_log['loss/loss_all'] = loss.item()

                if global_step % args.i_print == 0 or global_step < 10:

                    mse_error = img2mse(ret['rgb'], target_rgb).item()
                    scalars_to_log['train/psnr-training-batch'] = mse2psnr(mse_error)
                    logstr = '{} Epoch: {}  step: {} '.format(args.expname, epoch, global_step)
                    for k in scalars_to_log.keys():
                        logstr += ' {}: {:.6f}'.format(k, scalars_to_log[k])
                        writer.add_scalar(k, scalars_to_log[k], global_step)

                if global_step % args.i_weights == 0:
                    logger.info('Saving checkpoints at {} to {}...'.format(global_step, ckpt_folder))
                    fpath = os.path.join(ckpt_folder, 'model_{:06d}.pth'.format(global_step))
                    model.save_model(fpath)

                if global_step % args.i_img == 0:
                    logger.info('Logging a random validation view ...')
                    val_data = next(val_loader_iterator).squeeze()
                    log_view_to_tb(writer, global_step, val_data,
                                   model, reference_dataset, val_sample_kwargs,
                                   prefix='val/', depth_range=args.bounds
                                   )

                    torch.cuda.empty_cache()
                    logger.info('Logging current training view...')
                    train_view_idx = int(global_step // args.i_img - 1) % 3
                    train_data = test_dataset[args.train_list[train_view_idx]]
                    log_view_to_tb(writer, global_step, train_data,
                                   model, reference_dataset, val_sample_kwargs,
                                   prefix='train/', depth_range=args.bounds)
                    writer.flush()

                p_bar.set_postfix({'iter': '{}'.format(global_step), 'loss': '{0:1.5f}'.format(loss.item())})
                p_bar.update(1)

                global_step += 1
                if global_step > model.start_step + args.n_iters + 1:
                    break

            p_bar.close()
        epoch += 1
        logger.info('Epoch {} finished'.format(epoch))

    train_time = time.time() - train_start_time
    eval_train(args, model, test_dataset, reference_dataset, val_sample_kwargs, out_folder, global_step,
               writer=writer, save_image=True, reload=False, if_debug=False, train_time=train_time)


if __name__ == '__main__':
    args = config.load_config()
    train(args)
