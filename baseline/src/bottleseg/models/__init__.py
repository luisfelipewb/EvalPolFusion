# from __future__ import annotations
from typing import Callable, Dict

import torch.nn as nn
import segmentation_models_pytorch as smp


registry: Dict[str, Callable[..., nn.Module]] = {}


def register(name: str):
    """Decorator to register a model under a string key."""

    def _wrap(fn: Callable[..., nn.Module]):
        registry[name] = fn
        return fn

    return _wrap


def get_model(name: str, **kwargs) -> nn.Module:
    try:
        return registry[name](**kwargs)
    except KeyError as exc:
        raise ValueError(
            f"Unknown model '{name}'.  Available: {list(registry)}"
        ) from exc



@register("unet")
def _unet(classes: int = 1, **kw) -> nn.Module:  # kw passes extra args if needed
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=classes,
        activation=None,
        **kw,
    )


@register("unetpp")
def _unetpp(classes: int = 1, **kw) -> nn.Module:
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=classes,
        activation=None,
        **kw,
    )


@register("deeplab")
def _deeplab(classes: int = 1, **kw) -> nn.Module:
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=classes,
        activation=None,
        **kw,
    )


@register("pspnet")
def _pspnet(classes: int = 1, **kw) -> nn.Module:
    return smp.PSPNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        classes=classes,
        activation=None,
        **kw,
    )


@register("segformer")
def _segformer(classes: int = 1, variant: str = "b1", **kw) -> nn.Module:
    """
    variant – e.g. 'b1', 'b2', 'b3', etc.  See HF hub for allowed names.
    """
    from transformers import SegformerForSemanticSegmentation

    return SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/mit-{variant}",
        num_labels=classes,
        ignore_mismatched_sizes=True,
        **kw,
    )


