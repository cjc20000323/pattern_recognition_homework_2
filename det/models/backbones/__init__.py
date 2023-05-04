from .resnet import build_resnet, build_resnet_fpn


def build_backbone(cfg):
    name = cfg.MODEL.BACKBONE
    if name == "resnet":
        backbone = build_resnet(depth=cfg.MODEL.RESNET_DEPTH)
    elif name == "resnet-fpn":
        backbone = build_resnet_fpn(depth=cfg.MODEL.RESNET_FPN_DEPTH)
    else:
        raise ValueError("Unknown backbone: {}".format(name))
    return backbone