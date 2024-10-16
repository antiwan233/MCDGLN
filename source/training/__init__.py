from omegaconf import DictConfig
import torch
from typing import List
import logging
from torch_geometric.loader import DataLoader
from .training import Train
from MCDGLNTrain import MCDGLNTrain
from torch.optim import Optimizer
from source.components.lr_scheduler import LRScheduler


def training_factory(cfg: DictConfig,
                     model: torch.nn.Module,
                     optimizers:List[Optimizer],
                     lr_schedulers:  List[LRScheduler],
                     dataloaders: DataLoader,
                     logger: logging.Logger,
                     fold: int) -> Train:

    # config.model.get("train", None)的意思是从config.model中获取train的值，如果没有则返回None
    # 如果config.model中没有train的值，则使用config.training.name的值 "Train"
    # 即Class Train是一个基础类，而又可以根据config.model.train的值来选择不同的训练类
    # 此处的train是一个字符串，代表了一个类名
    train = cfg.model.get("train", None)
    if not train:
        train = cfg.training.name

    # return的一个例子是
    # return GCNTrain(*args)
    return eval(train)(cfg=cfg,
                       model=model,
                       optimizers=optimizers,
                       lr_schedulers=lr_schedulers,
                       dataloaders=dataloaders,
                       logger=logger,
                       fold=fold
                       )
