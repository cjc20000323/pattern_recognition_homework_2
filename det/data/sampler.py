import random

from torch.utils.data import Sampler


class BatchSampler(object):

    def __init__(self, base_sampler: Sampler, is_train: bool, cfg):
        self.base_sampler = base_sampler
        self.is_train = is_train
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.drop_last = cfg.TRAIN.DROP_LAST
        self.multiscale_step = cfg.TRAIN.MULTISCALE_STEP
        self.multiscale_period = cfg.TRAIN.MULTISCALE_PERIOD
        self.multiscale_range = cfg.TRAIN.MULTISCALE_RANGE
        self.multiscale_end_epoch = cfg.TRAIN.MULTISCALE_END_EPOCH
        self.enable_mosaic = cfg.TRAIN.ENABLE_MOSAIC
        self.mosaic_prob = cfg.TRAIN.MOSAIC_PROB
        self.mosaic_end_epoch = cfg.TRAIN.MOSAIC_END_EPOCH

        if self.is_train:
            self.base_h = cfg.TRAIN.INPUT_H
            self.base_w = cfg.TRAIN.INPUT_W
        else:
            self.base_h = cfg.TEST.INPUT_H
            self.base_w = cfg.TEST.INPUT_W
        self.input_h = self.base_h
        self.input_w = self.base_w
        self.current_batch = 0
        self.current_epoch = 0
    
    def __iter__(self):
        self.current_batch = 0
        self.current_epoch += 1
        batch = []
        for ind in self.base_sampler: 
            is_mosaic = self.is_train \
                and self.enable_mosaic \
                and self.current_epoch < self.mosaic_end_epoch \
                and (random.random() < self.mosaic_prob)
            batch.append([ind, is_mosaic, self.input_h, self.input_w])
            if len(batch) == self.batch_size:
                self.current_batch += 1
                yield batch
                self._update_input_size()
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.base_sampler) // self.batch_size
        else:
            return 1 + len(self.base_sampler) // self.batch_size
    
    def _update_input_size(self):
        if not self.is_train \
            or self.current_batch % self.multiscale_period != 0 \
            or self.current_epoch >= self.multiscale_end_epoch:
            return
        dH = random.randint(-self.multiscale_range, self.multiscale_range)
        dW = random.randint(-self.multiscale_range, self.multiscale_range)
        self.input_h = self.base_h + dH * self.multiscale_step
        self.input_w = self.base_w + dW * self.multiscale_step
