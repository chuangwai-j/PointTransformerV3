"""
Model Builder

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""
#from . import BACKBONES, ENCODERS, HEADS, LOSSES, MODELS
import copy
from pointcept.utils.registry import Registry

MODELS = Registry("models")
MODULES = Registry("modules")


def build_model(cfg):
    """Build models."""
    model = MODELS.build(cfg)
    return model
