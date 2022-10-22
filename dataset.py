import pickle
import numpy as np
import torch
import torch.utils.data

"""
对数据集进行提取
版本说明:
V1
时间: 2022.10.21
作者: fff
说明: RML2016a.pkl 数据集读取, onehot 类型标签制作
"""


class SigDataset(torch.utils.data.Dataset):
    """
    用于DataLoader进行数据加载
    参数说明:
        输入:
            X,Y: 训练时的样本数据和标签
    """

    def __init__(self, X, Y):
        self.data = X
        self.labels = Y
        # print(self.data.shape)
        # print(self.labels.shape)

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index])
        # mask = torch.topk(torch.from_numpy(self.labels[index]), 1)[1].squeeze(1)
        mask = self.labels[index]
        return image, mask

    def __len__(self):
        return len(self.labels)


def to_onehot(yy):
    """
    onehot 类型标签制作
    参数说明:
        输入:
            yy (list): size=(nsamples)

        输出:
            yy1 (narray): size=(nsamples, max(yy)+1)
    """
    # 下标从0开始，所以要+1
    yy1 = np.zeros([len(yy), max(yy) + 1])
    # np.arange(x) = 0:1:x
    yy1[np.arange(len(yy)), yy] = 1

    return yy1


def get_dataRML2016a(filename, seed, onehot_flag=True):
    """
    RML2016a.pkl 数据集读取
    参数说明:
        输入:
            filename: 文件路径
            seed: 随机参数
            onehot_flag (bool): 是否将标签转化为onehot类型，True:转化,False:不转化，默认为True

        输出:
            X_train,Y_train (narray): size=(nsamples, channels=2, ndots=128)
            Y_train,Y_test (narray): size=(nsamples, ) or (nsamples, len(mods)=11)
    """
    mods, snrs, X, lbls = get_keys_valuesRML2016a(filename)
    ########################################################
    X_train, X_test, Y_train, Y_test, _ = split_RML2016a(seed,  X, lbls, mods, onehot_flag)

    # 均转化为array
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


def get_keys_valuesRML2016a(filename):
    """
    获取数据集中的键值
    参数说明:
        输入:
            filename (string): 数据集地址

        输出:
            mods (list): 排序后的调制类型
            snrs (list): 排序后的信噪比 [-10, ..., 18]
            X (narray): 信号数据
            lbls (list): 所有调制信号对应的(mod, snr)
    """
    # 数据大小 11mods * 20snrs * 1000, Xd的类型为字典型{('mod',snr), value}，
    Xd = pickle.load(open(filename, 'rb'), encoding='latin')
    # len(Xd.keys())=11*20 Xd.keys()=('mod',snr)
    # 分别取出snrs和mods并进行排序，set用于删去重复项，map 表示对列表中每一元素共同操作
    snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1, 0])
    X = []
    lbls = []
    del snrs[0:5]  # 只用-10--18db

    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod, snr)])
            for i in range(Xd[(mod, snr)].shape[0]):
                lbls.append((mod, snr))
    X = np.array(X)
    X = np.vstack(X)  # 对数组进行垂直堆叠(165, 1000, 2, 128)--->(165000, 2, 128)

    return mods, snrs, X, lbls


def split_RML2016a(seed,  X, lbls, mods, onehot_flag):
    """
    对数据集进行拆分
    参数说明:
        输入:
            seed : 随机数种子
            X (narray): 信号数据
            lbls (list): 所有调制信号对应的(mod, snr)
            mods (list): 排序后的调制类型
            onehot_flag (bool): True 为 转为onehot类型

        输出:
            X_train,Y_train (narray): size=(nsamples, channels=2, ndots=128)
            Y_train,Y_test (narray): size=(nsamples, ) or (nsamples, len(mods)=11)
    """
    np.random.seed(seed)
    n_examples = X.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0, n_examples)) - set(train_idx))
    X_train = X[train_idx]
    X_test = X[test_idx]

    if onehot_flag:
        Y_train = to_onehot(list(map(lambda x: mods.index(lbls[x][0]), train_idx)))
        Y_test = to_onehot(list(map(lambda x: mods.index(lbls[x][0]), test_idx)))
    else:
        Y_train = list(map(lambda x: mods.index(lbls[x][0]), train_idx))
        Y_test = list(map(lambda x: mods.index(lbls[x][0]), test_idx))

    return X_train, X_test, Y_train, Y_test, test_idx


