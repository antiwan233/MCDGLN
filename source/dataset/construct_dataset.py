from omegaconf import DictConfig
from typing import Any, List, Tuple, Union
import numpy as np
from source.utils import retain_top_percent, get_hyperedges, sliding_windows
from torch_geometric.data import Data
from itertools import compress
import torch
from .load_abide import load_abide
from .load_mdd import load_mdd
from .load_adhd import load_adhd
from torch_geometric.utils import dense_to_sparse, add_self_loops


class MaskableList(list):
    def __getitem__(self, index):
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            return MaskableList(compress(self, index))


def threshing(corr, percent):

    threshold_corr = np.zeros_like(corr)

    for ind in range(corr.shape[0]):
        threshold_corr[ind] = retain_top_percent(corr[ind], percent)

    return threshold_corr


# 构造普通图数据集
def construct_graph_dataset(cfg: DictConfig) -> Tuple[List[Data], Any, Any]:

    # 获取dataset的name字段
    dataset_name = cfg.dataset.name

    # 根据dataset_name的值，调用不同的函数
    dataset_func = "load_" + dataset_name

    # eval函数执行字符串表达式，并返回表达式的值
    # 返回的是一个元组，包含了final_pearson, labels, site
    tc, corr, labels, sites = eval(dataset_func)(cfg.dataset.atlas)

    threshold_corr = threshing(corr, cfg.dataset.percent)

    # 将array转为tensor
    tc, threshold_corr, corr, labels = [torch.from_numpy(data).float() for data in (tc, threshold_corr, corr, labels)]

    # 构造Data对象
    graph_list = MaskableList([])
    for i in range(tc.shape[0]):
        edge_index, edge_attr = dense_to_sparse(threshold_corr[i])
        graph = Data(x=corr[i],
                     edge_index=edge_index,
                     edge_attr=edge_attr,
                     y=labels[i])
        graph_list.append(graph)

    return graph_list, labels, sites


# 构造超图数据集
def construct_hypergraph_dataset(cfg: DictConfig) -> Tuple[List[Data], Any, Any]:

    dataset_name = cfg.dataset.name

    dataset_func = "load_" + dataset_name

    tc, corr, labels, sites = eval(dataset_func)(cfg.dataset.atlas)

    # 首先通过保留top百分比的方式，去除FC中的噪声，有效。
    threshold_corr = threshing(corr, cfg.dataset.percent)

    # 将array转为tensor
    tc, corr, labels = [torch.from_numpy(data).float() for data in (tc, corr, labels)]

    hypergraph_list = MaskableList([])
    for i in range(tc.shape[0]):

        incidence_matrix = get_hyperedges(threshold_corr[i], cfg.thresh_counts, threshold_corr.shape[1])
        incidence_matrix = torch.Tensor(incidence_matrix).float()
        edge_index, edge_attr = dense_to_sparse(incidence_matrix)

        hyperedge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)

        hypergraph = Data(x=corr[i],
                          edge_index=edge_index,
                          edge_weight=hyperedge_weight,
                          y=labels[i],)
        hypergraph_list.append(hypergraph)

    return hypergraph_list, labels, sites


# 构造滑动窗口数据集
def construct_sliding_window_dataset(cfg: DictConfig) -> Tuple[List[Data], Any, Any]:

    dataset_name = cfg.dataset.name

    dataset_func = "load_" + dataset_name

    tc, corr, labels, sites = eval(dataset_func)(cfg.dataset.atlas)

    # 首先通过保留top百分比的方式，去除FC中的噪声，有效。
    # 如果要把这个作为嵌入特征的话，最好不要卡阈值，会丢失信息
    threshold_corr = threshing(corr, cfg.dataset.percent)

    windows = sliding_windows(tc[:, :, cfg.cut_timeseries_length],
                              cfg.model.window_size,
                              cfg.model.stride,
                              is_retain=True)

    windows, threshold_corr, labels = [torch.from_numpy(data).float()
                                       for data in (windows, threshold_corr, labels)]

    graph_list = MaskableList([])
    for i in range(tc.shape[0]):

        graph = Data(x=threshold_corr[i],
                     windows=windows[i],
                     y=labels[i])

        graph_list.append(graph)

    return graph_list, labels, sites
