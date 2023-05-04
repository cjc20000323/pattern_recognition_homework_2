import os

import cv2
import torch
import numpy as np
from loguru import logger

from configs import cfg
from models import Detector
from data.transform import ValTransform


# 这里暂时认为每张图像里只有一个人脸
def crop_single_img(model: torch.nn.Module, img_tensor: torch.Tensor,
                    ori_img: np.ndarray):
    model.eval()
    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0).to(cfg.DEVICE)
        bboxes_pred, categories_pred, scores_pred, other_preds = model(
            img_tensor, None, None, None
        )
        if len(scores_pred[0]) == 0:
            return None
        max_score_ind = torch.argmax(scores_pred[0])
        bbox = bboxes_pred[0][max_score_ind]
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = int(bbox[2])
        y1 = int(bbox[3])
        imh, imw = ori_img.shape[:2]
        if (not 0 <= x0 < imw) or (not 0 <= x1 < imw) or (not 0 <= y0 < imh) or (not 0 <= y1 < imh) \
            or (x1 <= x0) or (y1 <= y0):
            return None

        crop_res = ori_img[y0:y1, x0:x1]
        return crop_res


def crop_face(data_path, ckpt_path=None, recursive=True, 
              save=True, save_dir="crop"):
    cfg.freeze()
    model = Detector(cfg).to(cfg.DEVICE)
    transform = ValTransform(cfg)
    if save:
        os.makedirs(save_dir, exist_ok=True)

    def _is_img_file(path):
        if os.path.isfile(path) and os.path.splitext(path)[-1] in [".jpg", ".png", ".bmp"]:
            return True
        return False
    
    def _load_and_transform(img_path):
        ori_img = cv2.imread(img_path)
        if ori_img is None:
            return None, None
        inh, inw = ori_img.shape[:2]
        img_tensor, _, _, _ = transform(ori_img, None, None, inh, inw)
        return img_tensor, ori_img
    
    def _get_img_path_list(dir, recursive, last_items=[]):
        items = os.listdir(dir)
        img_path_list = []
        for item in items:
            item_path = os.path.join(dir, item)
            if _is_img_file(item_path):
                img_path_list.append((last_items, os.path.join(dir, item)))
            elif os.path.isdir(item_path) and recursive:
                img_path_list.extend(_get_img_path_list(item_path, recursive, [*last_items, item]))
        return img_path_list
    
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    failed = []
    if _is_img_file(data_path):
        img_tensor, ori_img = _load_and_transform(data_path)
        if img_tensor is not None:
            crop_res = crop_single_img(model, img_tensor, ori_img)
            if save:
                if crop_res is not None:
                    cv2.imwrite(os.path.join(save_dir, os.path.basename(data_path)), crop_res)
                else:
                    cv2.imwrite(os.path.join(save_dir, "det_failed_" + os.path.basename(data_path)), crop_res)
        else:
            failed.append(data_path)
    elif os.path.isdir(data_path):
        img_path_list = _get_img_path_list(data_path, True, [])
        __cnt = 0
        for dir_items, img_path in img_path_list:
            __cnt += 1
            print("\rProcessing {}/{}".format(__cnt, len(img_path_list)), end="")
            img_tensor, ori_img = _load_and_transform(img_path)
            if img_tensor is None:
                failed.append(img_path)
                continue
            crop_res = crop_single_img(model, img_tensor, ori_img)
            if save:
                os.makedirs(os.path.join(save_dir, *dir_items), exist_ok=True)
                img_fname = os.path.basename(img_path)
                if crop_res is not None:
                    cv2.imwrite(os.path.join(save_dir, os.path.join(*dir_items, img_fname)), crop_res)
                else:
                    cv2.imwrite(os.path.join(save_dir, os.path.join(*dir_items, "det_failed_" + img_fname)), ori_img)
        print("\n以下图片读取失败:")
        print(*failed, sep=", ")


if __name__ == "__main__":
    try:
        crop_face("../database/faces96",
                  "runs/resnet18-fpn/best.pth",
                  save_dir="faces96-crop")
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("Stopped by user")
        else:
            raise e