import cv2
import gin
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import matplotlib as mpl
from matplotlib import cm
from datetime import datetime
import os
import torch
import shutil
import numpy as np


HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def init_output_folder(args):
    ckpt_folder = os.path.join(args.ckpt_dir, args.expname)
    os.makedirs(ckpt_folder, exist_ok=True)

    ymd = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_folder = os.path.join(args.rootdir, args.expname, ymd)

    os.makedirs(out_folder, exist_ok=True)

    # copy gin config files
    if args.gin_configs:
        for gin_config in args.gin_configs:
            filename = os.path.basename(gin_config)
            f = os.path.join(out_folder, filename)
            if not os.path.isfile(f):
                shutil.copy(gin_config, f)

    return out_folder, ckpt_folder

def save_gin_config(out_folder):
    conf = gin.operative_config_str()

    args_path = os.path.join(out_folder, 'model_args.gin')
    with open(args_path, 'w') as f:
        f.write(conf)

mse2psnr = lambda x: -10. * np.log(x + TINY_NUMBER) / np.log(10.)


def img2mse(x, y, mask=None):
    '''
    :param x: img 1, [(...), 3]
    :param y: img 2, [(...), 3]
    :param mask: optional, [(...)]
    :return: mse score
    '''
    if mask is None:
        return torch.mean((x - y) * (x - y))
    else:
        return torch.sum((x - y) * (x - y) * mask.unsqueeze(-1)) / (torch.sum(mask) * x.shape[-1] + TINY_NUMBER)


def img2psnr(x, y, mask=None):
    return mse2psnr(img2mse(x, y, mask).item())

def colorize_np(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False, cbar_precision=2):
    '''
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    '''
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        # 计算数据的 1% 百分位数和 100% 百分位数。
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1. - mask)

    cbar = get_vertical_colorbar(h=x.shape[0], vmin=vmin, vmax=vmax, cmap_name=cmap_name, cbar_precision=cbar_precision)

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1]:, :] = cbar
        else:
            x_new = np.concatenate((x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1)
        return x_new
    else:
        return x_new
def get_vertical_colorbar(h, vmin, vmax, cmap_name='jet', label=None, cbar_precision=2):
    '''
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    '''
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                    norm=norm,
                                    ticks=tick_loc,
                                    orientation='vertical')

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize(x, cmap_name='jet', mask=None, range=None, append_cbar=False, cbar_in_image=False):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x

def contraction(x, sample_kwargs):
    if "scene_aabb" in sample_kwargs:
        aabb = sample_kwargs["scene_aabb"]
    else:
        border = float(sample_kwargs["border"])
        sign_list = [-1, -1, -1, 1, 1, 1]
        aabb = [float(border * item) for item in sign_list]
        aabb = torch.tensor(aabb, dtype=torch.float32).to(x.device)

    aabb_min, aabb_max = aabb[:3].unsqueeze(0), aabb[3:].unsqueeze(0)
    x = (x - aabb_min) / (aabb_max - aabb_min)

    return x