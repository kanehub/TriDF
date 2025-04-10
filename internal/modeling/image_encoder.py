import gin
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet34

@gin.configurable()
class ImageEncoder(nn.Module):
    def __init__(
        self,
        output_dim: int = 64
    ) -> None:
        super(ImageEncoder, self).__init__()
        self.resnet = resnet34(True)
        self.output_dim = output_dim

    def forward(self, x):
        if self.output_dim == 3:
            # 输出 rgb
            return x
        else:
            latents = self.query_network(x)

        latent_sz = latents[0].shape[-2:]
        for i in range(len(latents)):
            latents[i] = F.interpolate(
                latents[i], latent_sz, mode="bilinear", align_corners=True
            )

        latents = torch.cat(latents, dim=1)
        return latents

    def query_network(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        feats1 = self.resnet.relu(x)
        latents = [feats1]
        if self.output_dim == 64:
            return latents

        feats2 = self.resnet.layer1(self.resnet.maxpool(feats1))
        latents.append(feats2)

        if self.output_dim == 128:
            return latents

        feats3 = self.resnet.layer2(feats2)
        latents.append(feats3)
        if self.output_dim == 256:
            return latents

        feats4 = self.resnet.layer3(feats3)
        latents.append(feats4)
        if self.output_dim == 512:
            return latents
