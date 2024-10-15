import numpy as np
from numpy import ndarray


def get_hyperedges(x: ndarray,
                   thresh_counts: int,
                   num_nodes: int) -> ndarray:

    hyperedges = []

    co_occurrence = np.dot((x > 0).astype(int), (x > 0).astype(int).T)

    for i in range(co_occurrence.shape[0]):
        # 查找共现次数大于等于阈值的节点
        frequent_neighbors = np.where(co_occurrence[i] >= thresh_counts)[0]
        if len(frequent_neighbors) > 0:
            hyperedges.append(list(frequent_neighbors))

    filtered_hyperedges = [h for h in hyperedges if len(h) > 1]

    num_hyperedges = len(filtered_hyperedges)

    adj_matrix = np.zeros((num_nodes, num_hyperedges), dtype=int)

    rows = []
    cols = []

    for j, hyperedge in enumerate(filtered_hyperedges):
        rows.extend(hyperedge)
        cols.extend([j] * len(hyperedge))

    rows = np.array(rows)
    cols = np.array(cols)

    adj_matrix[rows, cols] = 1

    return adj_matrix
