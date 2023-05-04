from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms
import numpy as np


class CTHead(nn.Module):

    def __init__(
        self,
        cfg,
        in_channels: int,
        inp_downsample_ratio: int,
    ):
        super(CTHead, self).__init__()
        self.inp_downsample_ratio = inp_downsample_ratio
        self.cfg = cfg

        self.cls_branch = self._make_branch(in_channels, cfg.DATA.NUM_CLASSES,
                                            bias_fill=True, bias_val=-2.19)
        self.reg_branch = self._make_branch(in_channels, 2)
        self.wh_branch = self._make_branch(in_channels, 2)

    def _make_branch(self, in_channels: int, out_channels: int,
                     bias_fill: bool = False, bias_val: float = 0):
        conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        relu = nn.ReLU(inplace=True)
        conv2 = nn.Conv2d(64, out_channels, 1)
        if bias_fill:
            conv2.bias.data.fill_(bias_val)
        return nn.Sequential(conv1, relu, conv2)

    def forward(
        self,
        x: torch.Tensor,
        bboxes: List[torch.Tensor],
        categories: List[torch.LongTensor],
        img_infos: List[dict],
        **kwargs,
    ):
        cls_pred = torch.sigmoid(self.cls_branch(x))
        reg_pred = torch.sigmoid(self.reg_branch(x))
        wh_pred = torch.exp(self.wh_branch(x))

        if self.training:
            return self.get_losses(
                x.shape, bboxes, categories,
                cls_pred, reg_pred, wh_pred
            )
        else:
            bboxes_pred, categories_pred, scores_pred = self.decode(
                x.shape,
                cls_pred, reg_pred, wh_pred
            )
            return bboxes_pred, categories_pred, scores_pred, None


    def get_losses(
        self,
        feat_shape,
        bboxes: List[torch.Tensor],
        categories: List[torch.LongTensor],
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        wh_pred: torch.Tensor,
    ):
        B = len(bboxes)
        cls_targets, reg_targets, wh_targets, target_inds = self._gen_targets(
            feat_shape, bboxes, categories
        )
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        wh_pred = wh_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        
        cls_loss, reg_loss, wh_loss = 0., 0., 0.
        for i in range(B):
            cls_loss += self._calc_cls_loss(cls_pred[i], cls_targets[i])
            reg_loss += self._calc_reg_loss(reg_pred[i], reg_targets[i], target_inds[i])
            wh_loss += self._calc_wh_loss(wh_pred[i], wh_targets[i], target_inds[i])
        cls_loss /= B
        reg_loss /= B
        wh_loss /= B
        return {
            "total_loss": cls_loss + reg_loss + wh_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "wh_loss": wh_loss,
        }
    
    def decode(
        self,
        feat_shape,
        cls_pred: torch.Tensor,
        reg_pred: torch.Tensor,
        wh_pred: torch.Tensor,
    ):
        cls_pred_pooled = F.max_pool2d(cls_pred, kernel_size=3, stride=1, padding=1)
        keep = (cls_pred_pooled == cls_pred).float()
        cls_pred = cls_pred * keep

        B, _, featH, featW = feat_shape
        cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.cfg.DATA.NUM_CLASSES)
        reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        wh_pred = wh_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
        cls_pred, reg_pred, wh_pred = cls_pred.to("cpu"), reg_pred.to("cpu"), wh_pred.to("cpu")
        grid_y, grid_x = torch.meshgrid(
            torch.arange(featH),
            torch.arange(featW),
            indexing="ij"
        )
        grids = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)

        bboxes_pred, categories_pred, scores_pred = [], [], []
        for i in range(B):
            cls_pred_i = cls_pred[i]
            reg_pred_i = reg_pred[i]
            wh_pred_i = wh_pred[i]

            scores_i, categories_i = torch.max(cls_pred_i, dim=1)
            scores_i, topk_inds = torch.topk(scores_i, k=self.cfg.TEST.TOPK)
            topk_inds = topk_inds[scores_i > 0.20]
            scores_i = scores_i[scores_i > 0.20]
            
            categories_i = categories_i[topk_inds]
            reg_pred_i = reg_pred_i[topk_inds]
            wh_pred_i = wh_pred_i[topk_inds]

            ct_pred_i = grids[topk_inds] + reg_pred_i
            bboxes_pred_i = torch.zeros(len(topk_inds), 4)
            bboxes_pred_i[:, [0, 1]] = ct_pred_i - wh_pred_i / 2.0
            bboxes_pred_i[:, [2, 3]] = ct_pred_i + wh_pred_i / 2.0
            bboxes_pred_i = bboxes_pred_i * self.inp_downsample_ratio

            if self.cfg.TEST.WITH_NMS:
                keep = nms(bboxes_pred_i.to("cuda"), scores_i.to("cuda"), self.cfg.TEST.NMS_THRESH).to("cpu")
                bboxes_pred_i = bboxes_pred_i[keep]
                categories_i = categories_i[keep]
                scores_i = scores_i[keep]

            bboxes_pred.append(bboxes_pred_i)
            categories_pred.append(categories_i)
            scores_pred.append(scores_i)

        return bboxes_pred, categories_pred, scores_pred
                

    def _calc_cls_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ):
        gt = gt.to(pred.device)
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt, 4)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        pred = torch.clamp(pred, 1e-12)

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss
    
    def _calc_reg_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        inds: torch.Tensor,
    ):
        gt = gt.to(pred.device)
        pred = pred[inds]
        loss = F.l1_loss(pred, gt, reduction="sum")
        return loss / max(1, len(inds))

    def _calc_wh_loss(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        inds: torch.Tensor,
    ):
        gt = gt.to(pred.device)
        pred = pred[inds]
        loss = F.l1_loss(pred, gt, reduction="sum")
        return loss / max(1, len(inds))

    def _gen_targets(
        self, 
        feat_shape,
        bboxes: List[torch.Tensor],
        categories: List[torch.LongTensor],
    ):
        _, _, featH, featW = feat_shape
        cls_targets, reg_targets, wh_targets = [], [], []
        target_inds = []
        for bboxes_i, categories_i in zip(bboxes, categories):
            bboxes_i = bboxes_i / self.inp_downsample_ratio
            centers = (bboxes_i[:, 2:] + bboxes_i[:, :2]) / 2.0
            centers_int = centers.to(torch.int64) # 0: ct_x, 1: ct_y
            target_inds_i = centers_int[:, 1] * featW + centers_int[:, 0]
            wh_targets_i = bboxes_i[:, 2:] - bboxes_i[:, :2]
            reg_targets_i = centers - centers_int
            cls_targets_i = torch.zeros(self.cfg.DATA.NUM_CLASSES, featH, featW)
            self._generate_score_map(cls_targets_i, categories_i, wh_targets_i, 
                                     centers_int, self.cfg.MODEL.CT_HEAD_MIN_OVERLAP)
            cls_targets.append(cls_targets_i)
            reg_targets.append(reg_targets_i)
            wh_targets.append(wh_targets_i)
            target_inds.append(target_inds_i)
        return cls_targets, reg_targets, wh_targets, target_inds

    def _generate_score_map(self, fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = self._get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            self._draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    def _get_gaussian_radius(self, box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = torch.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = height + width
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))
    
    def _gaussian2D(self, radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m : m + 1, -n : n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    def _draw_gaussian(self, fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = self._gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top : y + bottom, x - left : x + right]
        masked_gaussian = gaussian[radius - top : radius + bottom, radius - left : radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top : y + bottom, x - left : x + right] = masked_fmap
