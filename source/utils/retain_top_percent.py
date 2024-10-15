import numpy as np
from numpy import ndarray


def retain_top_percent(matrix: ndarray,
                       percent: float = 0.2) -> ndarray:
    """
    保留对称矩阵中前percent的值，并确保输出仍然是对称的。

    :param matrix: 对称的 NumPy 数组
    :param percent: 要保留的百分比，默认为 0.2
    :return: 保留了前percent值的对称矩阵
    """
    # 提取上三角部分
    upper_triangle = np.triu(matrix, k=1)

    # 获取上三角部分的非零元素
    flat_upper = upper_triangle[upper_triangle != 0]

    # 计算阈值，保留前 20% 的值
    threshold = np.percentile(flat_upper, 100 - (100 * percent))

    # 保留大于阈值的元素
    upper_indices = np.where(upper_triangle >= threshold)

    out = np.zeros_like(matrix)

    # 更新矩阵
    for i, j in zip(*upper_indices):
        out[i, j] = out[j, i] = upper_triangle[i, j]

    return out
