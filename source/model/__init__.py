from omegaconf import DictConfig
from .test_model import MLP
from .GCN import GCN
from .Trans_Hyper import Trans_HyperGraph
from .MCDGLN import MCDGLN


def model_factory(cfg: DictConfig):

    return eval(cfg.model.name)(cfg=cfg)
