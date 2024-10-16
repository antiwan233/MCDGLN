from typing import List, Tuple, Any, Union
import numpy
import numpy as np
from omegaconf import DictConfig, open_dict
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from .load_abide import load_abide
from .load_mdd import load_mdd
from .load_adhd import load_adhd
from sklearn.model_selection import train_test_split
from .construct_dataset import construct_graph_dataset, construct_hypergraph_dataset
from .construct_dataset import construct_sliding_window_dataset


def dataset_factory(cfg: DictConfig) -> Tuple[List[Union[Data,]], Any, Any]:

    return eval("construct_" + cfg.data_type + "_dataset")(cfg)


def dataloader_factory(cfg: DictConfig,
                       dataset: List[Data],
                       labels: Any,
                       sites: Any,
                       train_index=None,
                       test_index=None) -> Tuple[DataLoader, DataLoader]:
    # create dataset
    # 此处的datasets是一个包含torch_geometric.data.Data对象的列表

    if cfg.cross_validation:
        print("Preparing Cross Validation DataLoader ... ")

        train_binary, test_binary = numpy.zeros(len(dataset), dtype=int), numpy.zeros(len(dataset), dtype=int)
        train_binary[train_index] = 1
        test_binary[test_index] = 1

        dataset_train, dataset_test, y_train, y_test = (dataset[train_binary],
                                                        dataset[test_binary],
                                                        labels[train_binary],
                                                        labels[test_binary])
        train_loader = DataLoader(dataset_train, batch_size=cfg.training.batch_size, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset_test, batch_size=cfg.training.batch_size, drop_last=True, pin_memory=True)

        print("Cross Validation DataLoader Done...")

        return train_loader, test_loader

    else:

        print("Preparing DataLoader ... ")

        dataset_train, dataset_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.1,
                                                                        random_state=cfg.seed, stratify=labels)

        train_loader = DataLoader(dataset_train, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(dataset_test, batch_size=cfg.training.batch_size, shuffle=True, drop_last=True, pin_memory=True)

        print("DataLoader Done...\n")

        return train_loader, test_loader
