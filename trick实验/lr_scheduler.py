from typing import List
from bisect import bisect_right

import torch


class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 1.0 / 3,
        warmup_epochs: int = 10,
        warmup_method: str = "linear",
        last_epoch: int = -1
    ) -> None:
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of increasing integers, got {}"
                .format(milestones)
            )
        
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant', 'linear' warmup_method accepted, got {}"
                .format(warmup_method)
            )
        
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    model = torch.nn.Linear(20, 10)
    scheduler = WarmupMultiStepLR(
        optimizer=torch.optim.SGD(model.parameters(), lr=3.5 * 1e-4),
        milestones=[40, 70],
        gamma=0.1,
        warmup_factor=0.1,
        warmup_epochs=10,
        warmup_method="linear",
        last_epoch=-1
    )

    lrs = []
    for epoch in range(120):
        scheduler.step(epoch=epoch)
        lr = scheduler.get_lr()[0]
        lrs.append(lr)
        
    plt.plot(list(range(120)), lrs)
    plt.show()

