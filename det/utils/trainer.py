import os
import time
import random
import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from .evaluator import Evaluator
from .metric import gpu_mem_usage, mem_usage, AverageMeter
from .visualize import draw_bboxes, draw_mask


__all__ = ["Trainer"]


class Trainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.evaluator = Evaluator(cfg)
        self.avg_meter = AverageMeter()
        self.cfg = cfg
        self.start_epoch = 1
        self.current_epoch = 0
        self.max_epochs = cfg.TRAIN.MAX_EPOCHS
        self.current_iter = 0
        self.n_iters_per_epoch = len(self.train_loader)
        self.best_ap = 0.0

    def run(self):
        self.before_train()

        if self.cfg.EVAL_ONLY:
            self._eval()
            return

        for _ in range(self.start_epoch, self.max_epochs + 1):
            self.before_epoch()
            for batch in self.train_loader:
                self.before_iter()
                self.train_step(batch)
                self.after_iter()
            self.after_epoch()
        self.after_train()

    def before_train(self):
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        ckpt_path = os.path.join(self.cfg.OUTPUT_DIR, "last.pth")
        self.model.to(self.cfg.DEVICE)
        if self.cfg.TRAIN.RESUME and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.cfg.DEVICE)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            self.start_epoch = ckpt["epoch"] + 1
            self.best_ap = ckpt["best_ap"]
            if self.cfg.EVAL_ONLY:
                logger.info("Evaluating the last ckpt")
            else:
                logger.info("Training from epoch {}".format(self.start_epoch))
        else:
            self.start_epoch = 1
            self.best_ap = 0.0
            if self.cfg.EVAL_ONLY:
                logger.info("Evaluating the init model")
            else:
                logger.info("Training from scratch")
        self.current_epoch = self.start_epoch - 1
        
    def before_epoch(self):
        self.current_iter = 0
        self.current_epoch += 1
        self.n_iters_per_epoch = len(self.train_loader)
        self.avg_meter.reset(keep=["time"])
        self.evaluator.reset()
        self.model.train()

    def before_iter(self):
        self.current_iter += 1
        self.optimizer.zero_grad()

    def train_step(self, batch):
        imgs, bboxes, categories, img_infos = batch
        imgs = imgs.to(self.cfg.DEVICE)

        start_time = time.time()
        loss_dict = self.model(
            imgs, bboxes, categories, img_infos
        )
        total_loss = loss_dict["total_loss"]
        total_loss.backward()
        self.optimizer.step()
        print(end='')
        self.avg_meter.update("time", time.time() - start_time)
        for k, v in loss_dict.items():
            self.avg_meter.update(k, v)

        if self.current_iter % self.cfg.TRAIN.LOG_INTERVAL == 0:
            info = "Epoch {} | Iter [{}/{}]".format(
                self.current_epoch, 
                self.current_iter, 
                self.n_iters_per_epoch
            )
            info += " | GPU mem {:.2f}MB | mem {:.2f}GB".format(
                gpu_mem_usage(), mem_usage()
            )
            for k, _ in loss_dict.items():
                info += " | {}: {:.3f}".format(k, self.avg_meter.avg_val(k))
            info += " | lr: {:.4e}".format(self.optimizer.param_groups[0]['lr'])
            imH, imW = batch[0].shape[2], batch[0].shape[3]
            info += " | HxW={}x{}".format(imH, imW)
            eta_seconds = ((self.max_epochs - self.current_epoch + 1) * self.n_iters_per_epoch \
                           - self.current_iter) * self.avg_meter.avg_val("time")
            info += " | ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(info)

    def after_iter(self):
        torch.cuda.empty_cache()
    
    def after_epoch(self):
        self.scheduler.step()
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.current_epoch,
            "best_ap": self.best_ap,
        }
        torch.save(ckpt, os.path.join(self.cfg.OUTPUT_DIR, "last.pth"))
        
        if self.current_epoch % self.cfg.TRAIN.EVAL_INTERVAL == 0:
            self.model.eval()
            ap50_95, ap50 = self._eval()
            if ap50_95 > self.best_ap:
                self.best_ap = ap50_95
                ckpt["best_ap"] = self.best_ap
                torch.save(ckpt, os.path.join(self.cfg.OUTPUT_DIR, "best.pth"))

    def after_train(self):
        return
    
    def _eval(self):
        self.model.eval()
        n_plot_img = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader):
                imgs, bboxes, categories, img_infos = batch
                imgs = imgs.to(self.cfg.DEVICE)

                self.evaluator.update_gt(bboxes, categories, img_infos)
                bboxes_pred, categories_pred, scores_pred, other_preds = self.model(
                    imgs, bboxes, categories, img_infos
                )
                if bboxes_pred is not None:
                    self.evaluator.update_dt(bboxes_pred, categories_pred,
                                             scores_pred, img_infos)
                if n_plot_img < 50 and random.random() < 0.4:
                    ind = random.randint(0, len(imgs) - 1)
                    img_file = os.path.join(
                        self.cfg.DATA.VAL_DIR, 
                        "images", 
                        img_infos[ind]["file_name"]
                    )
                    img_pad_left = img_infos[ind]["transform"]["pad_left"]
                    img_pad_top = img_infos[ind]["transform"]["pad_top"]
                    img_pad_right = img_infos[ind]["transform"]["pad_right"]
                    img_pad_bottom = img_infos[ind]["transform"]["pad_bottom"]
                    if bboxes_pred is not None:
                        bboxes_pred_i = bboxes_pred[ind]
                        bboxes_pred_i[:, [0, 2]] -= img_pad_left
                        bboxes_pred_i[:, [1, 3]] -= img_pad_top
                        bboxes_pred_i /= img_infos[ind]["transform"]["scale_ratio"]
                        img = draw_bboxes(img_file, bboxes_pred_i, categories_pred[ind],
                                          scores_pred[ind], self.cfg.DATA.CATEGORIES)
                        cv2.imwrite(os.path.join(self.cfg.OUTPUT_DIR, f"sample_pred_{n_plot_img}.jpg"), img)

                    if other_preds is not None:
                        if "mask_pred" in other_preds:
                            mask_pred = other_preds["mask_pred"][ind]
                            mask_pred = (mask_pred * 255).astype(np.uint8)
                            mask_pred = cv2.resize(mask_pred, (img_infos[ind]["width"], img_infos[ind]["height"]))
                            mask_pred = mask_pred[img_pad_top:(mask_pred.shape[0] - img_pad_bottom),
                                                  img_pad_left:(mask_pred.shape[1] - img_pad_right)]
                            img_mask = draw_mask(img_file, mask_pred)
                            cv2.imwrite(os.path.join(self.cfg.OUTPUT_DIR, f"sample_mask_pred_{n_plot_img}.jpg"), img_mask)

                    n_plot_img += 1
            ap50_95, ap50 = self.evaluator.evaluate()
        return ap50_95, ap50