from .ct_head import CTHead


def build_head(cfg, in_channels, inp_downsample_ratio):
    name = cfg.MODEL.HEAD
    if name == "ct-head":
        head = CTHead(cfg, in_channels, inp_downsample_ratio)
    else:
        raise ValueError("Unknown head: {}".format(name))
    return head
