import math
import random

import torch


class RandomErasing(object):
    """
    Randomly selects a rectangle region in an image and erases its pixels.

    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
    """

    def __init__(
        self,
        probability: float = 0.5,
        sl: float = 0.02,
        sh: float = 0.40,
        r1: float = 0.30
    ) -> None:
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img: torch.Tensor):
        if random.uniform(0, 1) >= self.probability:
            return img
        
        img_mean = img.mean((1, 2))
        for _ in range(100):
            area = img.shape[1] * img.shape[2]
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.shape[2] and h < img.shape[1]:
                top = random.randint(0, img.shape[1] - h)
                left = random.randint(0, img.shape[2] - w)
                if img.shape[0] == 3:
                    img[0, top:(top+h), left:(left+w)] = img_mean[0]
                    img[1, top:(top+h), left:(left+w)] = img_mean[1]
                    img[2, top:(top+h), left:(left+w)] = img_mean[2]
                else:
                    img[0, top:(top+h), left:(left+w)] = img_mean[0]
                return img
            
        return img