import numpy as np
import torch
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def sliding_windows(data: np.ndarray,
                    window_size: int,
                    stride: int):
    """
    :param data: 一般形状为N*V*T
    :param window_size:
    :param stride:
    :return:
    """

    N, V, T = data.shape

    # 步长为1的输出
    out_1 = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size, axis=2)

    # 步长为stride的输出
    out_stride = out_1[:, :, ::stride, :]

    # 存放输出的矩阵
    out_fc = np.zeros((N, out_stride.shape[2], V, V))

    # 循环多个时间窗内的FC
    for i in range(N):
        for j in range(out_stride.shape[2]):
            out_fc[i][j] = np.corrcoef(out_stride[i, :, j, :])

    # 处理计算FC后的nan、posinf、neginf
    out_fc = np.nan_to_num(out_fc, nan=0, posinf=1, neginf=-1)

    # print(True in np.isnan(out_fc))
    # print(True in np.isinf(out_fc))

    # 返回时间窗划分后的FC矩阵，形状应当为N*M*V*V, 其中M为时间窗的个数
    return out_fc
