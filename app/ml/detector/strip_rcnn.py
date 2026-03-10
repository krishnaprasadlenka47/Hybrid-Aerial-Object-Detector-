import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms, roi_align


class StripConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, strip_size: int = 7):
        super().__init__()
        self.horizontal = nn.Conv2d(
            in_channels, out_channels // 2,
            kernel_size=(1, strip_size), padding=(0, strip_size // 2)
        )
        self.vertical = nn.Conv2d(
            in_channels, out_channels // 2,
            kernel_size=(strip_size, 1), padding=(strip_size // 2, 0)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.horizontal(x)
        v = self.vertical(x)
        out = torch.cat([h, v], dim=1)
        return self.relu(self.bn(out))


class StripFPN(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64):
        super().__init__()
        self.stage1 = nn.Sequential(
            StripConvBlock(in_channels, base_channels, strip_size=7),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            StripConvBlock(base_channels, base_channels * 2, strip_size=7),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            StripConvBlock(base_channels * 2, base_channels * 4, strip_size=5),
            nn.MaxPool2d(2),
        )
        self.stage4 = nn.Sequential(
            StripConvBlock(base_channels * 4, base_channels * 8, strip_size=3),
            nn.MaxPool2d(2),
        )

        c = base_channels
        self.lateral3 = nn.Conv2d(c * 4, 256, 1)
        self.lateral4 = nn.Conv2d(c * 8, 256, 1)
        self.smooth3 = nn.Conv2d(256, 256, 3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, 3, padding=1)

    def forward(self, x: torch.Tensor):
        c1 = self.stage1(x)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)

        p4 = self.lateral4(c4)
        p3 = self.smooth3(
            self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        )
        p4 = self.smooth4(p4)

        return p3, p4


class RegionProposalHead(nn.Module):
    def __init__(self, in_channels: int = 256, num_anchors: int = 9):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.cls_head = nn.Conv2d(256, num_anchors * 2, 1)
        self.reg_head = nn.Conv2d(256, num_anchors * 4, 1)

    def forward(self, feat: torch.Tensor):
        x = F.relu(self.conv(feat))
        cls = self.cls_head(x)
        reg = self.reg_head(x)
        return cls, reg


class HybridDetectionHead(nn.Module):
    def __init__(self, roi_feat_dim: int = 256, vit_embed_dim: int = 768, num_classes: int = 15):
        super().__init__()
        self.roi_pool_size = 7
        fused_dim = roi_feat_dim * self.roi_pool_size * self.roi_pool_size + vit_embed_dim

        self.fusion_proj = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

        self.cls_head = nn.Linear(512, num_classes + 1)
        self.reg_head = nn.Linear(512, (num_classes + 1) * 4)

    def forward(self, roi_feats: torch.Tensor, vit_cls_token: torch.Tensor) -> tuple:
        B = roi_feats.size(0)
        roi_flat = roi_feats.view(B, -1)
        fused = torch.cat([roi_flat, vit_cls_token], dim=1)
        shared = self.fusion_proj(fused)
        cls_logits = self.cls_head(shared)
        bbox_deltas = self.reg_head(shared)
        return cls_logits, bbox_deltas


class HybridAerialDetector(nn.Module):
    def __init__(self, num_classes: int = 15, vit_embed_dim: int = 768, base_channels: int = 64):
        super().__init__()
        self.num_classes = num_classes

        self.fpn = StripFPN(in_channels=3, base_channels=base_channels)
        self.rpn = RegionProposalHead(in_channels=256, num_anchors=9)
        self.det_head = HybridDetectionHead(
            roi_feat_dim=256,
            vit_embed_dim=vit_embed_dim,
            num_classes=num_classes,
        )

    def forward(self, images: torch.Tensor, vit_features: torch.Tensor, proposals: torch.Tensor = None):
        p3, p4 = self.fpn(images)
        rpn_cls, rpn_reg = self.rpn(p3)

        vit_cls = vit_features[:, 0, :]

        if proposals is not None and len(proposals) > 0:
            roi_feats = roi_align(p3, [proposals], output_size=7, spatial_scale=1.0 / 8.0)
            cls_logits, bbox_deltas = self.det_head(roi_feats, vit_cls.repeat(roi_feats.size(0), 1))
        else:
            cls_logits = torch.zeros(1, self.num_classes + 1, device=images.device)
            bbox_deltas = torch.zeros(1, (self.num_classes + 1) * 4, device=images.device)

        return {
            "rpn_cls": rpn_cls,
            "rpn_reg": rpn_reg,
            "cls_logits": cls_logits,
            "bbox_deltas": bbox_deltas,
            "fpn_p3": p3,
            "fpn_p4": p4,
        }
