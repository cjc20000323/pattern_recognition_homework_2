import tempfile
import json
import contextlib
import io
import itertools
from typing import List

import torch
import numpy as np
from loguru import logger
from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


__all__ = ["Evaluator"]


def per_class_AR_table(coco_eval, class_names, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


class Evaluator:

    def __init__(self, cfg):
        self.cfg = cfg

        self.categories = []
        for id, name in enumerate(self.cfg.DATA.CATEGORIES):
            self.categories.append(
                {
                    "id": id,
                    "name": name,
                    "supercategory": name
                }
            )
        
        self.images = []
        self.annotations = []
        self.predictions = []

    def reset(self):
        self.images.clear()
        self.annotations.clear()
        self.predictions.clear()

    def update_gt(
        self, 
        bboxes: List[torch.Tensor], 
        categories: List[torch.Tensor],
        img_infos: List[dict],
    ):
        for bboxes_i, categories_i, img_info_i in zip(
            bboxes, categories, img_infos
        ):
            image_id = int(img_info_i["id"])
            img_item = {
                "id": image_id,
                "width": int(img_info_i["ori_width"]),
                "height": int(img_info_i["ori_height"]),
                "file_name": str(img_info_i["file_name"]),
            }
            self.images.append(img_item)

            for bbox, category_id in zip(bboxes_i, categories_i):
                bbox_left = float(bbox[0])
                bbox_top = float(bbox[1])
                bbox_right = float(bbox[2])
                bbox_bottom = float(bbox[3])
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top
                bbox_area = bbox_width * bbox_height

                ann_item = {
                    "id": len(self.annotations) + 1,
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "area": int(bbox_area),
                    "bbox": [int(bbox_left), int(bbox_top), int(bbox_width), int(bbox_height)],
                    "segmentation": [],
                    "iscrowd": 0,
                }
                self.annotations.append(ann_item)

    def update_dt(
        self,
        bboxes: List[torch.Tensor], 
        categories: List[torch.Tensor],
        scores: List[torch.Tensor],
        img_infos: List[dict],
    ):
        for bboxes_i, categories_i, scores_i, img_info_i in zip(
            bboxes, categories, scores, img_infos
        ):
            image_id = int(img_info_i["id"])
            for bbox, category_id, score in zip(
                bboxes_i, categories_i, scores_i
            ):
                bbox_left = float(bbox[0])
                bbox_top = float(bbox[1])
                bbox_right = float(bbox[2])
                bbox_bottom = float(bbox[3])
                bbox_width = bbox_right - bbox_left
                bbox_height = bbox_bottom - bbox_top
                pred_item = {
                    "image_id": image_id,
                    "category_id": int(category_id),
                    "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                    "score": float(score)
                }
                self.predictions.append(pred_item)

    def evaluate(self):
        if len(self.annotations) == 0 or len(self.predictions) == 0:
            return 0.0, 0.0
        
        gt_dict = {
            "images": self.images,
            "categories": self.categories,
            "annotations": self.annotations
        }
        _, tmp = tempfile.mkstemp()
        json.dump(gt_dict, open(tmp, "w"))
        cocoGt = COCO(tmp)
        _, tmp = tempfile.mkstemp()
        json.dump(self.predictions, open(tmp, "w"))
        cocoDt = cocoGt.loadRes(tmp)

        cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
        cocoEval.params.maxDets = [10, 100, 275]
        cocoEval.evaluate()
        cocoEval.accumulate()
        redirect_string = io.StringIO()
        with contextlib.redirect_stdout(redirect_string):
            cocoEval.summarize()
        info = "\n" + redirect_string.getvalue()
        cat_ids = list(cocoGt.cats.keys())
        cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
        if self.cfg.TEST.PER_CLASS_AP:
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "\nper class AP:\n" + AP_table + "\n"
        if self.cfg.TEST.PER_CLASS_AR:
            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "\nper class AR:\n" + AR_table + "\n"
        
        logger.info(info)
        ap50_95 = cocoEval.stats[0]
        ap50 = cocoEval.stats[1]
        return ap50_95, ap50