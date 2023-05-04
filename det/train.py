import os

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from loguru import logger

from configs import cfg
from data import Dataset, BatchSampler
from models import Detector
from utils import GradualWarmupScheduler, Trainer


def main():
    cfg.freeze()
    logger.add(os.path.join(cfg.OUTPUT_DIR, "train_log.txt"))

    train_dataset = Dataset(cfg, is_train=True)
    val_dataset = Dataset(cfg, is_train=False)
    train_sampler = BatchSampler(
        base_sampler=RandomSampler(train_dataset),
        is_train=True,
        cfg=cfg
    )
    val_sampler = BatchSampler(
        base_sampler=RandomSampler(val_dataset),
        is_train=False,
        cfg=cfg
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=Dataset.collate_fn,
        num_workers=cfg.DATA.NUM_WORKERS
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        collate_fn=Dataset.collate_fn,
        num_workers=cfg.DATA.NUM_WORKERS
    )

    model = Detector(cfg)
    optimizer = Adam(model.parameters(), lr=cfg.TRAIN.BASE_LR)
    after_scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=cfg.TRAIN.MAX_EPOCHS - cfg.TRAIN.WARMUP_EPOCHS, 
                                        eta_min=cfg.TRAIN.BASE_LR * cfg.TRAIN.WARMUP_MUL)
    scheduler = GradualWarmupScheduler(optimizer,
                                       cfg.TRAIN.WARMUP_MUL,
                                       cfg.TRAIN.WARMUP_EPOCHS,
                                       after_scheduler=after_scheduler)
    
    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, cfg)
    cudnn.benchmark = True

    logger.info("Config:\n" + str(cfg))
    trainer.run()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            logger.info("Stopped by user")
        else:
            raise e