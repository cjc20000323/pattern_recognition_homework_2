import psutil

import torch


__all__ = [
    "gpu_mem_usage",
    "mem_usage",
    "AverageMeter",
]

def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (MB).
    """
    mem_usage_bytes = torch.cuda.max_memory_allocated()
    return mem_usage_bytes / (1024 * 1024)


def mem_usage():
    """
    Compute the memory usage for the current machine (GB).
    """
    gb = 1 << 30
    mem = psutil.virtual_memory()
    return mem.used / gb


class AverageMeter:

    def __init__(self):
        self.sum_vals = dict()
        self.cnts = dict()

    def reset(self, keep=[]):
        keys = list(self.sum_vals.keys())
        for k in keys:
            if k not in keep:
                self.sum_vals.pop(k)
                self.cnts.pop(k)

    def update(self, name, val):
        if name not in self.sum_vals:
            self.sum_vals[name] = 0.0
            self.cnts[name] = 0
        self.sum_vals[name] += val
        self.cnts[name] += 1

    def avg_val(self, name):
        if name not in self.sum_vals:
            return 0
        return self.sum_vals[name] / self.cnts[name]