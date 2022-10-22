import numpy as np

"""
对数据集进行预处理
版本说明:
V1
时间: 2022.10.21
作者: fff
说明:  IQ信号转化为AP(amplitude,phase)信号, 对幅值信号归一化
"""


def to_amp_phase(X_train, X_test):
    """
    IQ信号转化为AP(amplitude,phase)信号
    参数说明:
        输入:
            X_train size: (nsamples, channels=2, ndots=128)  type: array
            X_test size: (nsamples, channels=2, ndots=128)  type: array

        输出:
            X_train size: (nsamples, channels=2, ndots=128)  type: array
            X_test size: (nsamples, channels=2, ndots=128)  type: array
    """
    # 转化为复信号
    X_train_cmplx = X_train[:, 0, :] + 1j * X_train[:, 1, :]
    X_test_cmplx = X_test[:, 0, :] + 1j * X_test[:, 1, :]
    print("X_train_cmplx:", X_train_cmplx.shape)  # X_train_cmplx: (nsamples, 128)

    # 复信号转化为幅值相位信号
    X_train_amp = np.abs(X_train_cmplx)
    X_test_amp = np.abs(X_test_cmplx)
    # 归一化
    X_train_amp = norm_pad_zeros(X_train_amp)
    X_test_amp = norm_pad_zeros(X_test_amp)

    # 复信号转化为相位信号并进行归一化
    X_train_ang = np.arctan2(X_train[:, 1, :], X_train[:, 0, :]) / np.pi
    X_test_ang = np.arctan2(X_test[:, 1, :], X_test[:, 0, :]) / np.pi

    _, ndots = X_train_amp.shape
    # 改变维度 (nsamples, ndots) ---> (nsamples, 1, ndots)
    X_train_amp = np.reshape(X_train_amp, (-1, 1, ndots))
    X_train_ang = np.reshape(X_train_ang, (-1, 1, ndots))
    X_test_amp = np.reshape(X_test_amp, (-1, 1, ndots))
    X_test_ang = np.reshape(X_test_ang, (-1, 1, ndots))

    # 将幅值和相位信号拼接 (nsamples, 2, ndots)
    X_train = np.concatenate((X_train_amp, X_train_ang), axis=1)
    X_test = np.concatenate((X_test_amp, X_test_ang), axis=1)

    return X_train, X_test


def norm_pad_zeros(X_train_amp):
    """
    对幅值信号归一化
    参数说明:
        输入:
            X_train_amp size: (nsamples, ndots=128)  type: array

        输出:
            X_train_amp size: (nsamples, ndots=128)  type: array
    """
    print("Pad:", X_train_amp.shape)
    for i in range(X_train_amp.shape[0]):
        # 对振值数据正则化
        # linalg = linear + algebra ,norm 表示范数，2范数
        X_train_amp[i, :] = X_train_amp[i, :] / np.linalg.norm(X_train_amp[i, :], 2)
    return X_train_amp



