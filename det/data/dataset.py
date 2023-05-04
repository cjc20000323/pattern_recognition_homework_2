import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset

from .transform import TrainTransform, ValTransform, mosaic


class Dataset(torchDataset):

    def __init__(self, cfg, is_train: bool = True):
        super(Dataset, self).__init__()

        self.data_dir = cfg.DATA.TRAIN_DIR if is_train else cfg.DATA.VAL_DIR
        self.ann_dir = os.path.join(self.data_dir, "annotations")
        self.img_dir = os.path.join(self.data_dir, "images")
        self.ann_files = os.listdir(self.ann_dir)
        self.transform = TrainTransform(cfg) if is_train else ValTransform(cfg)
        self.is_train = is_train

    def __len__(self):
        return len(self.ann_files)
    
    def __getitem__(self, index):
        ind, is_mosaic, input_h, input_w = index
        if self.is_train and is_mosaic:
            inds = [ind] + [random.randint(0, self.__len__() - 1) for _ in range(3)]
            img_l, bboxes_l, categories_l = [], [], []
            for ind in inds:
                img, bboxes, categories, img_info = self._pull_item(ind)
                img_l.append(img)
                bboxes_l.append(bboxes)
                categories_l.append(categories)
            img, bboxes, categories = mosaic(
                img_l, bboxes_l, categories_l, input_h, input_w
            )
            img_info["height"] = img.shape[0]
            img_info["width"] = img.shape[1]
        else:
            img, bboxes, categories, img_info = self._pull_item(ind)

        img, bboxes, categories, img_info = self.transform(
            img, bboxes, categories, input_h, input_w, img_info
        )
        bboxes_w = bboxes[:, 2] - bboxes[:, 0]
        bboxes_h = bboxes[:, 3] - bboxes[:, 1]
        keep = (bboxes_w > 1e-4) & (bboxes_h > 1e-4)
        bboxes = bboxes[keep]
        categories = categories[keep]
        return img, bboxes, categories, img_info

    def _pull_item(self, index: int):
        ann_file = self.ann_files[index]
        with open(os.path.join(self.ann_dir, ann_file), "r") as f:
            labels = f.readlines()
        bboxes, categories = [], []
        for label in labels:
            bbox_left, bbox_top, bbox_right, bbox_bottom, category \
                = map(int, label.strip().split(','))
            bboxes.append([bbox_left, bbox_top, bbox_right, bbox_bottom])
            categories.append(category)
        bboxes = np.array(bboxes, dtype=np.float32)
        categories = np.array(categories, dtype=np.int32)
        
        img_file_name = ann_file.replace(".txt", ".jpg")
        img = cv2.imread(os.path.join(self.img_dir, img_file_name))
        img_info = {
            "ori_height": img.shape[0],
            "ori_width": img.shape[1],
            "height": img.shape[0],
            "width": img.shape[1],
            "file_name": img_file_name,
            "id": index,
            "transform": {
                "scale_ratio": 1.0,
                "pad_left": 0,
                "pad_top": 0,
            }
        }
        return img, bboxes, categories, img_info
    
    @staticmethod
    def collate_fn(batch):
        items = list(zip(*batch))
        if isinstance(items[0][0], torch.Tensor):
            imgs = torch.stack(items[0], dim=0)
        elif isinstance(items[0][0], np.ndarray):
            imgs = np.stack(items[0], axis=0)
        else:
            raise ValueError
        
        bboxes = list(items[1])
        categories = list(items[2])
        img_infos = list(items[3])
        return imgs, bboxes, categories, img_infos