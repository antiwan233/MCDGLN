import numpy as np
import pandas as pd
import scipy.io as sio
import os
from sklearn.preprocessing import StandardScaler
import nilearn.connectome as connectome
import re
np.set_printoptions(threshold=np.inf)


def pick_filename(s):

    match = re.search(r'([A-Za-z0-9_]+)_\d+_rois_cc200.', s)
    if match:
        return match.group(1)
    else:
        return None


def load_abide(atlas='cc200', num_sample=1035):
    # 读取时间序列数据
    if num_sample == 1035:
        ts_file_path = f'/home/user/data/wangpeng/data/abide/1035/timeseries/{atlas}'
        csv_file = r'/home/user/data/wangpeng/data/abide/1035/abide_1035.csv'
    else:
        ts_file_path = f'/home/user/data/wangpeng/data/abide/timeseries/{atlas}'
        # 读取表型信息文件
        csv_file = r'/home/user/data/wangpeng/data/abide/abide_884.csv'

    file_list = os.listdir(ts_file_path)
    # 读取标签和站点信息
    if num_sample == 1035:

        labels = pd.read_csv(csv_file)['DX_GROUP'].values.squeeze()
        labels[labels == 2] = 0
        sites = pd.read_csv(csv_file)['SITE_ID'].values.squeeze()

        sorted_file_list = sorted(file_list)
        cmu27 = sorted_file_list[0:27]
        sorted_file_list[0:37] = sorted_file_list[27:64]
        sorted_file_list[37:64] = cmu27

    else:
        labels = pd.read_csv(csv_file)['Asd'].values.squeeze()
        sites = pd.read_csv(csv_file)['SITE_ID'].values.squeeze()

        # 改一下没排对的文件顺序
        sorted_file_list = sorted(file_list)
        cmu5 = sorted_file_list[0:5]
        sorted_file_list[0:37] = sorted_file_list[5:42]
        sorted_file_list[37:42] = cmu5

    # print(sorted_file_list)

    # 处理站点信息

    for site in sites:

        if site == 'LEUVEN_1' or site == 'LEUVEN_2':
            sites[sites == site] = 'LEUVEN'
        elif site == 'UCLA_1' or site == 'UCLA_2':
            sites[sites == site] = 'UCLA'
        elif site == 'UM_1' or site == 'UM_2':
            sites[sites == site] = 'UM'

    # 计算相关信息
    # print(set(sites))

    # 200是atlas定义的ROI数量
    # 400是手动定义的时间序列的填充长度
    # 必须要指定元素类型，不然会默认整型，向下取整

    nrois = np.loadtxt(os.path.join(ts_file_path, file_list[0])).shape[1]

    corr = np.zeros((len(file_list), nrois, nrois))
    tc = np.zeros((len(file_list), nrois, 400))
    high_order_corr = np.zeros((len(file_list), nrois, nrois))

    for i, file in enumerate(sorted_file_list):
        time_series = np.loadtxt(os.path.join(ts_file_path, file))

        time_series = time_series.astype(np.float32).transpose()

        correlation_matrix = np.corrcoef(time_series, rowvar=True)

        time_series = np.pad(time_series, ((0, 0), (0, 400 - time_series.shape[1])), 'constant', constant_values=0)

        tc[i] = time_series

        # print(tc[i]==time_series)
        # print(time_series)
        # print(tc[i])

        # 这里的自连接存在，即自相关系数为1，后期需要的话再去除
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)

        # 将对角线元素设置为0
        np.fill_diagonal(correlation_matrix, 0)

        corr[i] = correlation_matrix

        # 计算高阶相关

        # high_order_correlation = np.corrcoef(correlation_matrix, rowvar=True)
        # high_order_correlation = np.nan_to_num(high_order_correlation, nan=0, posinf=1, neginf=-1)
        # np.fill_diagonal(high_order_correlation, 0)
        #
        # high_order_corr[i] = high_order_correlation

    # 计算偏相关
    # conn_measure = connectome.ConnectivityMeasure(kind='partial correlation')
    # par_connectivity = conn_measure.fit_transform(tc.transpose((0, 2, 1)))
    # par_connectivity = np.nan_to_num(par_connectivity, nan=0, posinf=1, neginf=-1)
    #
    # # 去除自连接
    # for j in range(par_connectivity.shape[0]):
    #     np.fill_diagonal(par_connectivity[j], 0)

    return tc, corr, labels, sites


# tc, cor, par, labels, sites = load_abide('cc200')
