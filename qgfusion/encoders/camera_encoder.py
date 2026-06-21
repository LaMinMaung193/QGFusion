"""
Camera Encoder -- Stage 1 (ENCODER column, blue lane in diagram)

Backbone: ResNet-50 + FPN, pretrained on nuImages (recommended) or ImageNet (fallback).
Input:  multi-view RGB images, shape (B, N_cam, 3, H, W)
Output: multi-scale feature maps Fc = [P3, P4, P5], each (B, N_cam, C, H_i, W_i)

Risk level (per proposal Sec 7): LOW
"""
import torch
import torch.nn as nn
import torchvision


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super().__init__()
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list]
        )

    def forward(self, feats):
        # feats: low-res -> high-res NOT assumed; we top-down merge from last (lowest-res) to first
        laterals = [l_conv(f) for l_conv, f in zip(self.lateral_convs, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest"
            )
        return [o_conv(l) for o_conv, l in zip(self.output_convs, laterals)]


class CameraEncoder(nn.Module):
    """ResNet-50 + FPN camera encoder. Produces multi-scale Fc per camera view."""

    def __init__(self, out_channels: int = 256, pretrained_backbone: str = "imagenet", freeze_stages: int = 1):
        super().__init__()
        # "none" gives randomly-initialized weights -- useful for offline wiring tests
        # (e.g. tools/test_forward.py) where downloading pretrained weights isn't possible.
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone == "imagenet" else None
        backbone = torchvision.models.resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # stride 4,  256 ch
        self.layer2 = backbone.layer2  # stride 8,  512 ch
        self.layer3 = backbone.layer3  # stride 16, 1024 ch
        self.layer4 = backbone.layer4  # stride 32, 2048 ch

        self.fpn = FPN(in_channels_list=[512, 1024, 2048], out_channels=out_channels)

        # TODO(risk=LOW): swap pretrained_backbone="nuimages" to load an nuImages-pretrained
        # checkpoint (e.g. mmdetection3d's resnet50_nuim weights) instead of ImageNet weights.
        if pretrained_backbone == "nuimages":
            raise NotImplementedError(
                "Plug in an nuImages-pretrained ResNet-50 checkpoint here -- "
                "see README.md > 'Pretrained weights' for pointers."
            )

        if freeze_stages > 0:
            self._freeze(freeze_stages)

    def _freeze(self, n_stages):
        modules = [self.stem, self.layer1, self.layer2, self.layer3, self.layer4][: n_stages + 1]
        for m in modules:
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, N_cam, 3, H, W)
        Returns:
            Fc: list[Tensor] = [P3, P4, P5], each (B, N_cam, C, H_i, W_i)
        """
        B, N, C, H, W = images.shape
        x = images.view(B * N, C, H, W)

        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        fpn_feats = self.fpn([c3, c4, c5])  # P3, P4, P5

        Fc = [f.view(B, N, *f.shape[1:]) for f in fpn_feats]
        return Fc
