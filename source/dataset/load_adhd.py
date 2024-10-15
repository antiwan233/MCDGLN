import numpy as np
import pandas as pd
import scipy.io as sio
import os
from sklearn.preprocessing import StandardScaler
import nilearn.connectome as connectome
import re


def extract_numbers(s):

    numbers = re.search(r'\d+', s)  # 找出所有的数字序列
    return int(numbers.group())


def load_adhd(atlas):

    """
    本文件用于将时间序列数据转换为数据集，需要处理的是以下几类数据：

    1. 时间序列数据
        - 此处应当对时间序列数据进行z-score标准化，从而使其分布符合标准正态分布
        - 但是值得注意的是，在使用PCC公式计算FC的时候，对时间序列数据进行标准化与否并不影响结果
        - 也可以不在这里处理时序数据，可以在代码的预处理部分进行，这里还是输入原始的时序数据
    2. 功能连接数据
        - 计算的公式一般为Pearson相关系数（PCC）
        - 计算出的结果会有nan, posinf, neginf等值，需要进行处理
    3. 站点信息
        - 要注意的是，站点信息读取时候要注意方式
        - 同名但是带有后缀的站点，比如UM_1和UM_2，应当进行合并
    4. 疾病标签信息
    """

    # 读取时间序列数据

    ts_file_path = f'/home/user/data/wangpeng/data/adhd/timeseries/{atlas}'
    file_list = os.listdir(ts_file_path)

    # 读取表型信息文件

    csv_file = '/home/user/data/wangpeng/data/adhd/adhd_768.csv'

    # 读取标签和站点信息

    labels = pd.read_csv(csv_file)['DX'].values.squeeze()
    sites = pd.read_csv(csv_file)['Site'].values.squeeze()
    #
    #
    # """
    # 验证标签的正确性
    # """
    #
    # scanid = pd.read_csv(csv_file)['ScanDir ID'].values.squeeze()

    sorted_file_list = sorted(file_list)

    labels[labels != 0] = 1
    sites[sites == 'NYU '] = 'NYU'
    sites[sites == 'KKI '] = 'KKI'

    # print(np.loadtxt(os.path.join(ts_file_path, file_list[0]), dtype=np.str_)[1:, 2:].astype(np.float64).shape)

    # 200是atlas定义的ROI数量
    # 400是手动定义的时间序列的填充长度

    nrois = np.loadtxt(os.path.join(ts_file_path, file_list[0]), dtype=np.str_)[1:, 2:].astype(np.float64).shape[1]
    corr = np.zeros((len(file_list), nrois, nrois))
    tc = np.zeros((len(file_list), nrois, 400))

    for i, file in enumerate(sorted_file_list):

        time_series = np.loadtxt(os.path.join(ts_file_path, file), dtype=np.str_)[1:, 2:].astype(np.float32)
        time_series = time_series.astype(np.float32).transpose()

        correlation_matrix = np.corrcoef(time_series, rowvar=True)

        time_series = np.pad(time_series, ((0, 0), (0, 400 - time_series.shape[1])), 'constant', constant_values=0)

        tc[i] = time_series

        # 这里的自连接存在，即自相关系数为1，后期需要的话再去除
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0, posinf=1, neginf=-1)

        np.fill_diagonal(correlation_matrix, 0)

        corr[i] = correlation_matrix

    # 计算偏相关
    # 使用补0后的时间序列数据计算偏相关会和原始的有一点差距，但是不大，可以接受
    # conn_measure = connectome.ConnectivityMeasure(kind='partial correlation')
    # par_connectivity = conn_measure.fit_transform(tc.transpose((0, 2, 1)))
    # par_connectivity = np.nan_to_num(par_connectivity, nan=0, posinf=1, neginf=-1)
    #
    # # 去除自连接
    # for j in range(par_connectivity.shape[0]):
    #     np.fill_diagonal(par_connectivity[j], 0)

    return tc, corr, labels, sites

# tc, cor, par, labels, sites = load_adhd('aal')
