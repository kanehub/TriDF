import numpy as np
import torch

from internal.render_ray import render_rays

def sample_rays_np(H, W, f, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i - W * .5 + 0.5) / f, -(j - H * .5 + 0.5) / f, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d # [H,W,3]

def render_patches(net,ref_dataset,render_step_size,nf_depth,
                         c2w, f, img_size, device,
                         patch_size, patch_origin):

    # set the rays from patch
    rays_o, rays_d = sample_rays_np(img_size, img_size, f, c2w)
    rays_o = rays_o[patch_origin[0]:patch_origin[0]+patch_size, patch_origin[1]:patch_origin[1]+patch_size]
    rays_d = rays_d[patch_origin[0]:patch_origin[0]+patch_size, patch_origin[1]:patch_origin[1]+patch_size]

    rays_o = torch.tensor(rays_o, device=device)
    rays_d = torch.tensor(rays_d, device=device)
    img_lines = list()
    for i in range(rays_d.shape[0]):
        rays_od = (rays_o[i], rays_d[i])
        img_lines.append(render_rays(net, rays_od, ref_dataset, render_step_size, near_far=nf_depth))

    ret = {}
    for key in img_lines[0].keys():
        ret[key] = torch.cat([img[key].unsqueeze(dim=0) for img in img_lines], dim=0).squeeze()
    return ret

def render_single_image(
        net, rays,  ref_dataset, sample_kwargs,
        ):
    '''
    rays_o, rays_d: [H,W,3]
    '''
    rays_o, rays_d = rays
    img_lines = list()
    # render per row
    for i in range(rays_d.shape[0]):
        rays_od = (rays_o[i], rays_d[i])
        img_lines.append(render_rays(net, rays_od, ref_dataset,sample_kwargs))
    ret = {}
    # concat the results
    for key in img_lines[0].keys():
        if key in ['rgb', 'acc', 'depth']:
            ret[key] = torch.cat([img[key].unsqueeze(dim=0) for img in img_lines], dim=0).squeeze()
    return ret

def render_single_image_w_chunks(net,rays, ref_dataset, sample_kwargs, chunk_size, heatmap=None):
    '''
    rays_o, rays_d: [H,W,3]
    '''
    rays_o, rays_d = rays
    img_lines = list()
    row_batch = chunk_size//rays_d.shape[1]
    # render per row
    for i in range(0,rays_d.shape[0],row_batch):
        rays_od = (rays_o[i:i+row_batch].reshape(-1,3), rays_d[i:i+row_batch].reshape(-1,3))
        img_lines.append(render_rays(net, rays_od, ref_dataset, sample_kwargs,heatmap=heatmap))
    ret = {}
    for key in img_lines[0].keys():
        if key in ['rgb', 'acc', 'depth']:
            dim_v = img_lines[0][key].shape[-1]
            ret[key] = torch.cat([img[key].unsqueeze(dim=0).reshape(row_batch,-1,dim_v) for img in img_lines], dim=0).squeeze()

    return ret


if __name__ == "__main__":
    a = [1,2,3,4,5,6,7]

    for i in range(0,7,3):
        print(a[i:i+3])