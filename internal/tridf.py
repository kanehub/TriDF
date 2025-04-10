import gin
import math
from typing import Callable
import torch
from torch import Tensor, nn
import tinycudann as tcnn

from internal.triplane.tri_mip import TriMipEncoding
from internal.triplane.activations import trunc_exp


@gin.configurable()
class TriDF(nn.Module):
    def __init__(
        self,
        ref_feat_dim: int,
        n_levels: int = 1,
        plane_size: int = 512,
        feature_dim: int = 8,
        geo_feat_dim: int = 16,
        net_depth_density = 8,
        net_width_density = 512,
        net_depth_base: int = 2,
        net_depth_color: int = 4,
        net_width: int = 128,
        med_feat_dim: int = 15,
        density_activation: Callable = lambda x: trunc_exp(x - 1),
    ) -> None:
        super().__init__()
        self.plane_size = plane_size
        self.log2_plane_size = math.log2(plane_size)
        self.geo_feat_dim = geo_feat_dim
        self.ref_feat_dim = ref_feat_dim
        self.med_feat_dim = med_feat_dim
        self.density_activation = density_activation


        self.encoding = TriMipEncoding(n_levels, plane_size, feature_dim)

        self.position_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 6,
            },
        )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.mlp_density = tcnn.Network(
            n_input_dims=self.position_encoding.n_output_dims + ref_feat_dim * 3,
            n_output_dims=1 + self.med_feat_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width_density,
                "n_hidden_layers": net_depth_density,
            },
        )

        self.mlp_base = tcnn.Network(
            n_input_dims=self.encoding.dim_out + self.med_feat_dim,
            n_output_dims=geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_base,
            },
        )
        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + geo_feat_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": net_width,
                "n_hidden_layers": net_depth_color,
            },
        )

    def query_density(
            self, x: Tensor, ref_feat: Tensor, return_feat: bool = False
    ):
        selector = ((x > 0.0) & (x < 1.0)).all(dim=-1)
        pos = self.position_encoding(x.view(-1, x.shape[-1]))

        # ref_feature
        ref_feat = ref_feat.permute(2, 0, 1, 3).view(-1, self.ref_feat_dim * 3)

        pos = torch.cat([pos, ref_feat], dim=-1)
        x = (
            self.mlp_density(pos)
            .view(list(x.shape[:-1]) + [1+self.med_feat_dim])
            .to(x)
        )



        density_before_activation, density_mlp_out = torch.split(
            x, [1, self.med_feat_dim], dim=-1
        )
        density = (
                self.density_activation(density_before_activation)
                * selector[..., None]
        )
        return {
            "density": density,
            "feature": density_mlp_out if return_feat else None,
        }

    def query_rgb(self, x, dir, embedding, level_vol: Tensor = None,):
        level = (
            level_vol if level_vol is None else level_vol + self.log2_plane_size
        )

        enc = self.encoding(
            x.view(-1, 3),
            level=level.view(-1, 1) if level is not None else level
        )

        enc = torch.cat([enc, embedding.view(-1, self.med_feat_dim)], dim=-1)


        base_mlp_out = (
            self.mlp_base(enc)
            .view(list(x.shape[:-1]) + [self.geo_feat_dim])
            .to(x)
        )

        # dir in [-1,1]
        dir = (dir + 1.0) / 2.0  # SH encoding must be in the range [0, 1]
        d = self.direction_encoding(dir.view(-1, dir.shape[-1]))
        h = torch.cat([d, base_mlp_out.view(-1, self.geo_feat_dim)], dim=-1)
        rgb = (
            self.mlp_head(h)
            .view(list(x.shape[:-1]) + [3])
            .to(x)
        )
        return {"rgb": rgb}

