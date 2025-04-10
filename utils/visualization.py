import torch
from internal.render_image import render_single_image
from utils import img2psnr,colorize
from evaluation.eval import ssim_fn


def log_view_to_tb(writer, global_step, ray_chunks,
                   model, reference_dataset, sample_kwargs,
                   prefix='',depth_range=None
                   ):
    model.switch_to_eval()

    with torch.no_grad():
        rays_o, rays_d, rgb_gt = torch.chunk(ray_chunks, 3, dim=-1)
        rays_od = (rays_o, rays_d)
        ret = render_single_image(
            model.net, rays_od, reference_dataset, sample_kwargs
        )

    psnr_curr_img = img2psnr(ret['rgb'], rgb_gt)

    rgb_im = ret['rgb'].detach().cpu()
    depth_im = ret['depth'].detach().cpu()
    depth_im_np = depth_im.numpy()
    depth_im = colorize(depth_im, cmap_name='jet', append_cbar=True, range=depth_range)

    rgb_gt = rgb_gt.detach().cpu()
    ssim_curr_img = ssim_fn(rgb_im.numpy(), rgb_gt.numpy())

    writer.add_image(prefix + 'rgb_pred', rgb_im, global_step, dataformats='HWC')
    writer.add_image(prefix + 'depth_pred', depth_im, global_step, dataformats='HWC')

    writer.add_scalar(prefix + 'psnr_image', psnr_curr_img, global_step)
    writer.add_scalar(prefix + 'ssim_image', ssim_curr_img, global_step)

    model.switch_to_train()
