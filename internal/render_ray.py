import torch
from torch import Tensor
from typing import Optional
import nerfacc
from nerfacc import OccupancyGrid, render_weight_from_density, accumulate_along_rays
from typing import Union, List, Optional, Callable

from utils import contraction


def render_rays(
    # scene
    radiance_field,
    rays,
    ref_feature,
    sample_kwargs: dict,
    # optional
    render_step_size: float = 1e-3,
    occupancy_grid: Optional[OccupancyGrid] = None,
    scene_aabb: Optional[torch.Tensor] = None,
    heatmap = None,
    near_far = None,
    near_plane: Optional[float] = None,
    far_plane: Optional[float] = None,
    render_bkgd: Optional[torch.Tensor] = None,
    cone_angle: float = 0.0,
    alpha_thre: float = 0.0,
):
    rays_o, rays_d = rays

    def sigma_fn(
            t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
    ) -> Tensor:
        """ Query density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation density values. (n_samples, 1).
        """
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        pts = t_origins[..., None, :] + t_dirs[..., None, :] * (t_starts[..., None] + t_ends[..., None]) / 2.0
        # positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        # shape = ray_indices.shape
        f_projection = ref_feature.feature_matching(pts)

        pts = contraction(pts, sample_kwargs)

        # level_vol = torch.tensor(0)
        # sigmas = radiance_field.query_density(pts, f_projection, level_vol)['density']
        sigmas = radiance_field.query_density(pts, f_projection)['density']
        sigmas = sigmas.squeeze(-1)

        return sigmas  # (n_samples, 1)

    def rgb_sigma_fn(
            t_starts: Tensor, t_ends: Tensor, ray_indices: Tensor
    ):
        """ Query rgb and density values from a user-defined radiance field.
        :params t_starts: Start of the sample interval along the ray. (n_samples, 1).
        :params t_ends: End of the sample interval along the ray. (n_samples, 1).
        :params ray_indices: Ray indices that each sample belongs to. (n_samples,).
        :returns The post-activation rgb and density values.
            (n_samples, 3), (n_samples, 1).
        """
        t_origins = rays_o[ray_indices]  # (n_samples, 3)
        t_dirs = rays_d[ray_indices]  # (n_samples, 3)
        pts = t_origins[..., None, :] + t_dirs[..., None, :] * (t_starts[..., None] + t_ends[..., None]) / 2.0
        # positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
        f_projection = ref_feature.feature_matching(pts)

        pts = contraction(pts, sample_kwargs)

        # level_vol = torch.tensor(0)
        # res = radiance_field.query_density(
        #     x=pts,
        #     ref_feat=f_projection,
        #     level_vol=level_vol,
        #     return_feat=True,
        # )
        # sigmas, feature = res['density'], res['feature']
        # rgbs = radiance_field.query_rgb(dir=t_dirs, embedding=feature)['rgb']

        res = radiance_field.query_density(
            x=pts,
            ref_feat=f_projection,
            return_feat=True,
        )
        sigmas, feature = res['density'], res['feature']
        rgbs = radiance_field.query_rgb(
            x=pts,
            dir=t_dirs,
            embedding=feature,
        )['rgb']

        # rgbs = torch.sum(rgbs, dim=-2)
        sigmas = sigmas.squeeze(-1)
        rgbs = rgbs.squeeze(-2)

        if heatmap is not None:
            if radiance_field.training:
                heatmap(pts, "train")
            else:
                heatmap(pts, "val")
        return rgbs, sigmas  # (n_samples, 3), (n_samples, 1)

    t_near = None
    t_far = None
    # 优先使用 near far
    if "near_far" in sample_kwargs:
        near_far = sample_kwargs["near_far"]
        render_step_size = sample_kwargs["render_step_size"]
        if near_far is not None:
            t_near, t_far = near_far[:,0].squeeze(), near_far[:,1].squeeze()
    elif "occupancy_grid" in sample_kwargs:
        occupancy_grid = sample_kwargs["occupancy_grid"]
        render_step_size = sample_kwargs["render_step_size"]
        scene_aabb = sample_kwargs["scene_aabb"]
    else:
        raise ValueError("Please provide near_far or occupancy_grid!")


    with torch.no_grad():
        ray_indices, t_starts, t_ends = nerfacc.ray_marching(
            rays_o, rays_d, sigma_fn=sigma_fn,
            render_step_size=render_step_size,
            alpha_thre=alpha_thre,
            scene_aabb=scene_aabb, grid=occupancy_grid,
            stratified=radiance_field.training,
            t_min=t_near, t_max=t_far
        )

    # Differentiable Volumetric Rendering.
    # rgb: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
    # rgb, opacity, depth = nerfacc.rendering(
    #     t_starts, t_ends, ray_indices,
    #     n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
    # )

    return rendering(
            t_starts,
            t_ends,
            ray_indices,
            rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )


def rendering(
    # ray marching results
    t_starts: torch.Tensor,
    t_ends: torch.Tensor,
    ray_indices: torch.Tensor,
    rays,
    # radiance field
    rgb_sigma_fn: Callable = None,  # rendering options
    render_bkgd: Optional[torch.Tensor] = None,
) :
    rays_o, rays_d = rays
    n_rays = rays_o.shape[0]
    # Query sigma/alpha and color with gradients
    rgbs, sigmas = rgb_sigma_fn(t_starts, t_ends, ray_indices.long())

    # Rendering
    weights = render_weight_from_density(
        t_starts,
        t_ends,
        sigmas,
        ray_indices=ray_indices,
        n_rays=n_rays,
    )

    # Rendering: accumulate rgbs, opacities, and depths along the rays.
    colors = accumulate_along_rays(
        weights, ray_indices=ray_indices, values=rgbs, n_rays=n_rays
    )
    opacities = accumulate_along_rays(
        weights, values=None, ray_indices=ray_indices, n_rays=n_rays
    )
    opacities.clamp_(
        0.0, 1.0
    )  # sometimes it may slightly bigger than 1.0, which will lead abnormal behaviours

    depths = accumulate_along_rays(
        weights,
        ray_indices=ray_indices,
        values=(t_starts + t_ends) / 2.0,
        n_rays=n_rays,
    )
    # depths = (
    #     depths * rays.ray_cos
    # )  # from distance to real depth (z value in camera space)



    # Background composition.
    if render_bkgd is not None:
        colors = colors + render_bkgd * (1.0 - opacities)

    ret = {
        'rgb': colors,
        'acc': opacities,
        'depth': depths,
        'density': sigmas,
        't_mid': (t_starts + t_ends) / 2.0,
        'weight': weights,
        'indices': ray_indices,
    }
    return ret



