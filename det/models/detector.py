from typing import List

import torch
import torch.nn as nn

from .backbones import build_backbone
from .heads import build_head


class Detector(nn.Module):

    def __init__(self, cfg):
        super(Detector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.head = build_head(
            cfg=cfg,
            in_channels=self.backbone.feat_channels,
            inp_downsample_ratio=self.backbone.downsample_ratio,
        )
        self.cfg = cfg

    def forward(
        self, 
        imgs: torch.Tensor,
        bboxes: List[torch.Tensor] = None,
        categories: List[torch.Tensor] = None,
        img_infos: List[dict] = None,
        **kwargs,
    ):
        feats = self.backbone(imgs)
        return self.head(feats, bboxes, categories, img_infos, **kwargs)
