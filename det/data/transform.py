import math
import random

import cv2
import numpy as np
import torch


def augment_hsv(img, hgain=5, sgain=30, vgain=30):
    hsv_augs = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain]
    hsv_augs *= np.random.randint(0, 2, 3)
    hsv_augs = hsv_augs.astype(np.int16)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_augs[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_augs[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_augs[2], 0, 255)

    img = cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR)
    return img


def get_aug_params(value, center=0):
    if isinstance(value, float) or isinstance(value, int):
        return random.uniform(center - value, center + value)
    elif len(value) == 2:
        return random.uniform(value[0], value[1])
    else:
        raise ValueError(
            "Affine params should be either a sequence containing two values\
             or single float values. Got {}".format(value)
        )


def get_affine_matrix(
    theight,
    twidth,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    # Rotation and Scale
    angle = get_aug_params(degrees)
    scale = get_aug_params(scales, center=1.0)

    if scale <= 0.0:
        raise ValueError("Argument scale should be positive")

    R = cv2.getRotationMatrix2D(angle=angle, center=(0, 0), scale=scale)

    M = np.ones([2, 3])
    # Shear
    shear_x = math.tan(get_aug_params(shear) * math.pi / 180)
    shear_y = math.tan(get_aug_params(shear) * math.pi / 180)

    M[0] = R[0] + shear_y * R[1]
    M[1] = R[1] + shear_x * R[0]

    # Translation
    translation_x = get_aug_params(translate) * twidth  # x translation (pixels)
    translation_y = get_aug_params(translate) * theight  # y translation (pixels)

    M[0, 2] = translation_x
    M[1, 2] = translation_y

    return M, scale


def apply_affine_to_bboxes(bboxes, theight, twidth, M, scale):
    num_gts = len(bboxes)

    # warp corner points
    corner_points = np.ones((4 * num_gts, 3))
    corner_points[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
        4 * num_gts, 2
    )  # x1y1, x2y2, x1y2, x2y1
    corner_points = corner_points @ M.T  # apply affine transform
    corner_points = corner_points.reshape(num_gts, 8)

    # create new boxes
    corner_xs = corner_points[:, 0::2]
    corner_ys = corner_points[:, 1::2]
    new_bboxes = (
        np.concatenate(
            (corner_xs.min(1), corner_ys.min(1), corner_xs.max(1), corner_ys.max(1))
        )
        .reshape(4, num_gts)
        .T
    )

    # clip boxes
    new_bboxes[:, 0::2] = new_bboxes[:, 0::2].clip(0, twidth - 1)
    new_bboxes[:, 1::2] = new_bboxes[:, 1::2].clip(0, theight - 1)

    bboxes[:, :4] = new_bboxes

    return bboxes


def random_affine(
    img,
    bboxes,
    theight=512, 
    twidth=800,
    degrees=10,
    translate=0.1,
    scales=0.1,
    shear=10,
):
    M, scale = get_affine_matrix(theight, twidth, degrees, translate, scales, shear)

    img = cv2.warpAffine(img, M, dsize=(twidth, theight), borderValue=(114, 114, 114))

    # Transform label coordinates
    if len(bboxes) > 0:
        bboxes = apply_affine_to_bboxes(bboxes, theight, twidth, M, scale)

    return img, bboxes


def horizontal_mirror(img, bboxes):
    _, width, _ = img.shape
    img = img[:, ::-1]
    bboxes[:, 0::2] = width - bboxes[:, 2::-2]
    return img, bboxes


def resize(img, bboxes, target_h, target_w, padding=False, letterbox=False):
    imH, imW = img.shape[0], img.shape[1]
    r = min(target_h / imH, target_w / imW)
    resized_img = cv2.resize(
        img, (int(r * imW), int(r * imH)),
        interpolation=cv2.INTER_LINEAR
    )
    if bboxes is not None:
        bboxes *= r
    if padding:
        result_img = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        if letterbox:
            pad_left = (target_w - resized_img.shape[1]) // 2
            pad_top = (target_h - resized_img.shape[0]) // 2
            result_img[pad_top:(pad_top + resized_img.shape[0]),
                    pad_left:(pad_left + resized_img.shape[1])] \
                = resized_img
            if bboxes is not None:
                bboxes[:, [0, 2]] += pad_left
                bboxes[:, [1, 3]] += pad_top
        else:
            pad_left, pad_top = 0, 0
            result_img[:resized_img.shape[0], :resized_img.shape[1]] \
                = resized_img
        pad_right = target_w - pad_left - resized_img.shape[1]
        pad_bottom = target_h - pad_top - resized_img.shape[0]
        return result_img, bboxes, r, pad_left, pad_top, pad_right, pad_bottom
    else:
        return resized_img, bboxes, r, 0, 0, 0, 0


def mosaic(img_l, bboxes_l, categories_l, input_h, input_w):
    mosaic_img = np.full((input_h, input_w, 3), 114, dtype=np.uint8)
    mosaic_bboxes, mosaic_categories = [], []
    ct_left = random.randint(input_w // 3, input_w // 3  *2)
    ct_top = random.randint(input_h // 3, input_h // 3 * 2)
    for i in range(4):
        img = img_l[i]
        bboxes = bboxes_l[i]
        categories = categories_l[i]

        if i == 0:
            left, top = 0, 0
            cut_w = ct_left
            cut_h = ct_top
        elif i == 1:
            left, top = 0, ct_top
            cut_w = ct_left
            cut_h = input_h - ct_top
        elif i == 2:
            left, top = ct_left, 0
            cut_w = input_w - ct_left
            cut_h = ct_top
        elif i == 3:
            left, top = ct_left, ct_top
            cut_w = input_w - ct_left
            cut_h = input_h - ct_top
            
        img, bboxes, _, _, _, _, _ = resize(
            img, bboxes, input_h, input_w, padding=True, letterbox=True
        )
        cut_left = random.randint(0, img.shape[1] - cut_w - 1)
        cut_top = random.randint(0, img.shape[0] - cut_h - 1)
        cut_right = cut_left + cut_w - 1
        cut_bottom = cut_top + cut_h - 1
        inter_bboxes = np.zeros_like(bboxes)
        inter_bboxes[:, 0] = np.where(bboxes[:, 0] >= cut_left, bboxes[:, 0], cut_left)
        inter_bboxes[:, 1] = np.where(bboxes[:, 1] >= cut_top, bboxes[:, 1], cut_top)
        inter_bboxes[:, 2] = np.where(bboxes[:, 2] <= cut_right, bboxes[:, 2], cut_right)
        inter_bboxes[:, 3] = np.where(bboxes[:, 3] <= cut_bottom, bboxes[:, 3], cut_bottom)
        inter_w = inter_bboxes[:, 2] - inter_bboxes[:, 0]
        inter_h = inter_bboxes[:, 3] - inter_bboxes[:, 1]
        keep = (inter_w >= 4) & (inter_h >= 4)
            
        bboxes = inter_bboxes[keep]
        categories = categories[keep]
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - cut_left + left
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - cut_top + top

        cut_img = img[cut_top:(cut_top + cut_h), cut_left:(cut_left + cut_w)]
        mosaic_img[top:(top + cut_h), left:(left + cut_w)] = cut_img
        mosaic_bboxes.append(bboxes)
        mosaic_categories.append(categories)

    mosaic_bboxes = np.concatenate(mosaic_bboxes, axis=0)
    mosaic_categories = np.concatenate(mosaic_categories, axis=0)
    return mosaic_img, mosaic_bboxes, mosaic_categories


def to_tensor(img, bboxes, categories, mean, std):
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = (img - mean) / std
    img = torch.FloatTensor(img)
    if bboxes is not None:
        bboxes = torch.FloatTensor(bboxes)
    if categories is not None:
        categories = torch.LongTensor(categories)
    return img, bboxes, categories


class TrainTransform:
    def __init__(self, cfg):
        self.cfg = cfg
        mean = np.array(self.cfg.DATA.TRANSFORM_MEAN)
        std = np.array(self.cfg.DATA.TRANSFORM_STD)
        self.mean = mean[:, np.newaxis, np.newaxis]
        self.std = std[:, np.newaxis, np.newaxis]

    def __call__(
        self, 
        img, 
        bboxes, 
        categories,
        input_h: int, 
        input_w: int,
        img_info = None,
    ):
        if random.random() < self.cfg.DATA.TRANSFORM_HSV_PROB:
            img = augment_hsv(
                img=img,
                hgain=self.cfg.DATA.TRANSFORM_HGAIN,
                sgain=self.cfg.DATA.TRANSFORM_SGAIN,
                vgain=self.cfg.DATA.TRANSFORM_VGAIN
            )
        if random.random() < self.cfg.DATA.TRANSFORM_HFLIP_PROB:
            img, bboxes = horizontal_mirror(img, bboxes)

        img, bboxes, r, pad_left, pad_top, pad_right, pad_bottom = resize(
            img, bboxes, input_h, input_w,
            padding=self.cfg.DATA.TRANSFORM_RESIZE_PADDING, 
            letterbox=self.cfg.DATA.TRANSFORM_RESIZE_LETTERBOX
        )
        img, bboxes = random_affine(
            img=img,
            bboxes=bboxes,
            theight=img.shape[0],
            twidth=img.shape[1],
            degrees=self.cfg.DATA.TRANSFORM_AFFINE_DEGREES,
            translate=self.cfg.DATA.TRANSFORM_AFFINE_TRANSLATE,
            scales=self.cfg.DATA.TRANSFORM_AFFINE_SCALES,
            shear=self.cfg.DATA.TRANSFORM_AFFINE_SHEAR
        )

        if img_info is not None:
            img_info["height"] = img.shape[0]
            img_info["width"] = img.shape[1]
            img_info["transform"]["scale_ratio"] = r
            img_info["transform"]["pad_left"] = pad_left
            img_info["transform"]["pad_top"] = pad_top
            img_info["transform"]["pad_right"] = pad_right
            img_info["transform"]["pad_bottom"] = pad_bottom

        if self.cfg.DATA.TRANSFORM_TO_TENSOR:
            img, bboxes, categories = to_tensor(
                img, bboxes, categories, self.mean, self.std
            )
        return img, bboxes, categories, img_info


class ValTransform:
    def __init__(self, cfg):
        self.cfg = cfg
        mean = np.array(self.cfg.DATA.TRANSFORM_MEAN)
        std = np.array(self.cfg.DATA.TRANSFORM_STD)
        self.mean = mean[:, np.newaxis, np.newaxis]
        self.std = std[:, np.newaxis, np.newaxis]

    def __call__(
        self, 
        img, 
        bboxes, 
        categories,
        input_h: int, 
        input_w: int,
        img_info = None,
    ):
        img, bboxes, r, pad_left, pad_top, pad_right, pad_bottom = resize(
            img, bboxes, input_h, input_w,
            padding=self.cfg.DATA.TRANSFORM_RESIZE_PADDING, 
            letterbox=self.cfg.DATA.TRANSFORM_RESIZE_LETTERBOX
        )

        if img_info is not None:
            img_info["height"] = img.shape[0]
            img_info["width"] = img.shape[1]
            img_info["transform"]["scale_ratio"] = r
            img_info["transform"]["pad_left"] = pad_left
            img_info["transform"]["pad_top"] = pad_top
            img_info["transform"]["pad_right"] = pad_right
            img_info["transform"]["pad_bottom"] = pad_bottom

        if self.cfg.DATA.TRANSFORM_TO_TENSOR:
            img, bboxes, categories = to_tensor(
                img, bboxes, categories, self.mean, self.std
            )
        return img, bboxes, categories, img_info
