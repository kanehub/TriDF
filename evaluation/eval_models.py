import sys

sys.path.append('./')

import imageio
import json
import os
import torch
import lpips
import time
import numpy as np

from datetime import datetime
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from internal.render_image import render_single_image_w_chunks
from utils import img2psnr, colorize_np
from internal.data_loaders import dataset_dict
from internal.feat_project import Projector
from internal.model import TriDFModel
from internal.smoothness import Smoother
from internal.sampling_strategy import SamplerType, TestSampler, TrainSampler

from utils import (img2mse, init_output_folder, mse2psnr, save_gin_config)


def ssim_fn(x, y):
    return structural_similarity(x, y, channel_axis=2, data_range=1.)


def lpips_fn(x, y, loss_fn):
    # RGB
    # transfer to CHW
    x = x.permute([2, 0, 1])
    y = y.permute([2, 0, 1])
    x = 2 * x - 1
    y = 2 * y - 1  # normalized to [0,1]

    return loss_fn(x, y)


def init_output_folder(args):
    ymd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = os.path.join(args.rootdir, args.expname, ymd)

    os.makedirs(out_folder, exist_ok=True)

    return out_folder


def main(
        args,
        writer=None,
        heatmap=None,
        save_image=True,
        if_debug=False,
        train_time=None,
        img_folder='rendering'
):
    iteration = os.path.basename(args.ckpt_path).split('.')[0].split('_')[-1]

    # output folder
    output_folder = init_output_folder(args)
    test_out_dir = os.path.join(output_folder, 'test')
    os.makedirs(test_out_dir, exist_ok=True)
    if save_image:
        pred_img_dir = os.path.join(test_out_dir, img_folder)
        os.makedirs(pred_img_dir, exist_ok=True)

    print("saving result to {}".format(test_out_dir))

    # create model

    model = TriDFModel(args, load_opt=not args.no_load_opt, load_scheduler=not args.no_load_scheduler)
    model.args.no_reload = False
    if args.ckpt_path is None:
        ckpt_folder = os.path.join(args.ckpt_dir, args.expname)
        model.load_from_ckpt(ckpt_folder, load_opt=False, load_scheduler=False)
    else:
        model.load_from_ckpt(out_folder=None, load_opt=False, load_scheduler=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_dataset = dataset_dict[args.dataset](args, 'train')

    test_dataset = dataset_dict[args.dataset](args, 'test')

    train_images = train_dataset.images
    train_images = torch.tensor(train_images, device=device, dtype=torch.float32).permute(0, 3, 1, 2)

    with torch.no_grad():
        featmaps = model.feature_net(train_images)
    reference_dataset = Projector(train_dataset, featmaps, device)

    # Ray Sampler
    sampler = TrainSampler(args, device, reference_dataset.img_size)
    val_sampler = TestSampler(args, device, reference_dataset.img_size)
    val_sample_kwargs = val_sampler()

    save_gin_config(output_folder)

    depth_regularizer = Smoother(args, test_dataset, reference_dataset)

    lpips_net = lpips.LPIPS(net='vgg').to(device)

    # metrics
    sum_test_psnr = 0
    sum_train_psnr = 0
    sum_test_ssims = 0
    sum_train_ssims = 0
    sum_test_lpips = 0
    sum_train_lpips = 0

    results_dict = {'train': {}, 'test': {}}

    test_loader = DataLoader(test_dataset, batch_size=1)
    total_num = len(test_loader)

    model.switch_to_eval()
    rendering_time = 0
    for i, test_data in enumerate(test_loader):
        time0 = time.time()
        with torch.no_grad():
            rays_o, rays_d, rgb_gt = torch.chunk(test_data.squeeze(), 3, dim=-1)
            rays_od = (rays_o, rays_d)
            ret = render_single_image_w_chunks(
                model.net, rays_od, reference_dataset, val_sample_kwargs, chunk_size=args.chunk_size, heatmap=heatmap
            )
        rgb_pred, depth_pred, acc_pred = ret['rgb'], ret['depth'], ret['acc']
        img_psnr = img2psnr(rgb_pred, rgb_gt)
        img_lpips = lpips_fn(rgb_pred, rgb_gt, lpips_net).item()
        rgb_pred_np = np.clip(rgb_pred.cpu().numpy(), 0, 1)
        rgb_gt_np = np.clip(rgb_gt.cpu().numpy(), 0, 1)
        img_ssim = ssim_fn(rgb_pred_np, rgb_gt_np)

        # saving outputs ...
        if save_image:
            rgb_pred = rgb_pred.detach().cpu()
            coarse_err_map = torch.sum((rgb_pred - rgb_gt.detach().cpu()) ** 2, dim=-1).numpy()
            coarse_err_map_colored = (colorize_np(coarse_err_map, range=(0., 1.)) * 255).astype(np.uint8)
            imageio.imwrite(os.path.join(pred_img_dir, 'err_map_{:03d}.png'.format(i)),
                            coarse_err_map_colored)

            rgb_pred_im = (255 * np.clip(rgb_pred.numpy(), a_min=0, a_max=1.)).astype(np.uint8)
            imageio.imwrite(os.path.join(pred_img_dir, 'color_{:03d}.png'.format(i)), rgb_pred_im)

            depth_pred_im = depth_pred.detach().cpu()
            imageio.imwrite(os.path.join(pred_img_dir, 'depth_{:03d}.png'.format(i)),
                            (depth_pred_im.numpy().squeeze() * 100.).astype(np.uint16))  # 0-65536

            depth_pred_colored = colorize_np(depth_pred_im, range=args.bounds)
            imageio.imwrite(os.path.join(pred_img_dir, 'vis_depth_{:03d}.png'.format(i)),
                            (255 * depth_pred_colored).astype(np.uint8))

        if i in args.train_list:
            view_type = 'train'
            sum_train_psnr += img_psnr
            sum_train_ssims += img_ssim
            sum_train_lpips += img_lpips
        else:
            view_type = 'test'
            sum_test_psnr += img_psnr
            sum_test_ssims += img_ssim
            sum_test_lpips += img_lpips
        print("==================\n"
              "curr_id: {} , {} view\n"
              "current psnr: {:03f}\n"
              "current ssim: {:03f}\n"
              "current lpips: {:03f}\n"
              "===================\n"
              .format(i, view_type, img_psnr, img_ssim, img_lpips))

        results_dict[view_type][i] = {'psnr': img_psnr, 'ssim': img_ssim, 'lpips': img_lpips}
        dt = time.time() - time0
        rendering_time += dt
        print('each image rendering time {:.05f} seconds with chunk size {}'.format(dt, args.chunk_size))
        if if_debug:
            if i == 3:
                break

    mean_test_psnr = sum_test_psnr / (total_num - len(args.train_list))
    mean_test_ssim = sum_test_ssims / (total_num - len(args.train_list))
    mean_test_lpips = sum_test_lpips / (total_num - len(args.train_list))

    mean_train_psnr = sum_train_psnr / len(args.train_list)
    mean_train_ssim = sum_train_ssims / len(args.train_list)
    mean_train_lpips = sum_train_lpips / len(args.train_list)

    results_dict['test_mean_psnr'] = mean_test_psnr
    results_dict['test_mean_ssim'] = mean_test_ssim
    results_dict['test_mean_lpips'] = mean_test_lpips

    results_dict['train_mean_psnr'] = mean_train_psnr
    results_dict['train_mean_ssim'] = mean_train_ssim
    results_dict['train_mean_lpips'] = mean_train_lpips

    results_dict['rendering_time'] = f'{rendering_time / total_num : .4f} s'
    if train_time is not None:
        print(f"Training time: {train_time} seconds")
        int_time = int(train_time)
        results_dict['train_time'] = f'{int_time // 60} min {int_time % 60} s'

    with open("{}/result_{}_{}.json".format(test_out_dir, args.scene_id, iteration), 'w') as f:
        json.dump(results_dict, f, indent=4, default=convert_to_builtin_type)

    if writer is not None:
        for k, v in results_dict.items():
            if 'mean' in k:
                tag = k.split('_')[0] + '/' + k.split('_')[-1]
                writer.add_text(tag, str(v))


def convert_to_builtin_type(obj):
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    raise TypeError("Object of type {} is not JSON serializable".format(type(obj)))


import config

if __name__ == "__main__":
    args = config.load_config()
    main(args)
