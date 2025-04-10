import sys
sys.path.append('./')
import os.path
import cv2
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

import config
from internal.render_image import render_single_image_w_chunks
from internal.data_loaders import dataset_dict
from internal.sampling_strategy import TestSampler
from internal.model import TriDFModel
from internal.feat_project import Projector
from utils import img2psnr,colorize



def render_path(args, work_path):

    save_path = os.path.join(work_path, args.scene_id, "images")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.torch_seed)
    np.random.seed(args.np_seed)
    torch.cuda.manual_seed(args.torch_cuda_seed)

    train_dataset = dataset_dict[args.dataset](args, 'train')
    test_dataset = dataset_dict[args.dataset](args, 'render_track')
    test_loader = DataLoader(test_dataset, batch_size=1)


    # create model
    model = TriDFModel(args, load_opt=False, load_scheduler=False)

    train_images = train_dataset.images
    train_images = torch.tensor(train_images, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
    # reference dataset
    with torch.no_grad():
        featmaps = model.feature_net(train_images)
    reference_dataset = Projector(train_dataset, featmaps, device)

    val_sampler = TestSampler(args, device, reference_dataset.img_size)
    val_sample_kwargs = val_sampler()

    model.switch_to_eval()
    idx = 0
    with tqdm(total=len(test_loader), desc="Rendered", ncols=85) as p_bar:
        for test_data in test_loader:
            with torch.no_grad():
                test_data = test_data.squeeze()
                rays_o, rays_d = torch.chunk(test_data, 2, dim=-1)
                rays_od = (rays_o, rays_d)
                ret = render_single_image_w_chunks(
                    model.net, rays_od,
                    reference_dataset, val_sample_kwargs,
                    chunk_size=4096
                )

                rgb_im = ret['rgb'].detach().cpu()
                depth_im = ret['depth'].detach().cpu()
                depth_im = colorize(depth_im, cmap_name='jet', append_cbar=False, range=args.bounds)

                rgb_np = rgb_im.numpy()
                rgb_np = np.clip(rgb_np, 0, 1)
                rgb_np = (rgb_np * 255).astype(np.uint8)
                rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

                depth_np = depth_im.numpy()
                depth_np = np.clip(depth_np, 0, 1)
                depth_np = (depth_np * 255).astype(np.uint8)
                depth_np = cv2.cvtColor(depth_np, cv2.COLOR_RGB2BGR)

                # cv2.imshow('rgb', rgb_np)
                # cv2.imshow('depth', depth_np)
                # cv2.waitKey(0)

                cv2.imwrite(os.path.join(save_path, 'rgb_{:03d}.png'.format(idx)), rgb_np)
                cv2.imwrite(os.path.join(save_path, 'depth_{:03d}.png'.format(idx)), depth_np)

            p_bar.set_postfix({'No.': '{}'.format(idx)})
            p_bar.update(1)
            idx += 1


import skvideo.io
def generate_video(workpath, scene_id, mode, fps=24):
    '''
    mode: 'rgb' or 'depth' or 'both'
    '''
    img_folder = os.path.join(workpath, scene_id, 'images')
    if mode == 'rgb' or mode == 'depth':

        output_video = f'{mode}.mp4'
        output_video_path = os.path.join(workpath, scene_id, output_video)
        images = [img for img in os.listdir(img_folder) if mode in img]
        images.sort()
        # 读取第一张图像以获取图像尺寸
        frame = cv2.imread(os.path.join(img_folder, images[0]))
        height, width, layers = frame.shape
        # 定义视频编解码器并创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用mp4v编解码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))


        for image in images:
            frame = cv2.imread(os.path.join(img_folder, image))
            out.write(frame)
        out.release()
    elif mode == 'both':
        output_video = 'rgb_depth.mp4'
        output_video_path = os.path.join(workpath, scene_id, output_video)
        rgb_images = [img for img in os.listdir(img_folder) if 'rgb' in img]
        depth_images = [img for img in os.listdir(img_folder) if 'depth' in img]
        rgb_images.sort()
        depth_images.sort()
        # 读取第一张图像以获取图像尺寸
        frame_rgb = cv2.imread(os.path.join(img_folder, rgb_images[0]))
        frame_depth = cv2.imread(os.path.join(img_folder, depth_images[0]))
        height_1, width_1, _ = frame_rgb.shape
        height_2, width_2, _ = frame_depth.shape

        assert height_1 == height_2, 'height not match'
        # 定义视频编解码器并创建VideoWriter对象
        fourcc = cv2.VideoWriter_fourcc(*'H264')  # 使用mp4v编解码器
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width_1 + width_2, height_1))
        for rgb, depth in zip(rgb_images, depth_images):
            frame_rgb = cv2.imread(os.path.join(img_folder, rgb))
            frame_depth = cv2.imread(os.path.join(img_folder, depth))
            frame = np.concatenate((frame_rgb, frame_depth), axis=1)
            out.write(frame)
        out.release()
    else:
        raise ValueError('mode should be rgb or depth or both')

import os
import subprocess as sp

def transfer_mp4v2h264(work_path, scene_id):


    # MPEG-4视频目录
    src_dir = os.path.join(work_path, scene_id)

    # H264视频目录
    dst_dir = os.path.join(work_path, scene_id, 'H264')

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for name in os.listdir(src_dir):
        if not '.mp4' in name:
            continue
        src_video_path = os.path.join(src_dir, name)
        dst_video_path = os.path.join(dst_dir, name)

        cmd = f'ffmpeg -i {src_video_path} -c:v libx264 -c:a copy -movflags +faststart {dst_video_path}'
        print(f'cmd: {cmd}')
        p = sp.Popen(cmd, shell=True)
        p.wait()

import shutil
def move_mp4(src_path, dst_path, mode):
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    for i in range(16):
        scene_id = f'scene_{i:03d}'
        src_video = os.path.join(src_path, scene_id, 'H264', f'{mode}.mp4')
        dst_video = os.path.join(dst_path, f'{i:03d}.mp4')
        if os.path.exists(src_video):
            shutil.copy(src_video, dst_video)

def observe_video_encode_type(video_path):

    # 打开视频文件

    cap = cv2.VideoCapture(video_path)

    # 检查视频文件是否成功打开
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # 获取视频的编码格式
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

    print(f"Video codec: {fourcc_str}")

    # 释放资源
    cap.release()


if __name__ == '__main__':
    args = config.load_config()
    ckpt_path = rf'.\checkpoint\{args.scene_id}\model_030000.pth'
    work_path = r'./exp/rendered'
    args.no_reload = False # 加载权重
    args.ckpt_path = ckpt_path

    render_path(args, work_path)
    generate_video(work_path, args.scene_id, 'rgb', fps=16)
    transfer_mp4v2h264(work_path, args.scene_id)