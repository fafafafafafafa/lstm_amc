import matplotlib.pyplot as plt
import numpy as np
import itertools
import csv
"""
绘图工具
版本说明:
V1
时间: 2022.10.22
作者: fff
说明: plot_confusion_matrix, plot_line_chart
"""


def plot_confusion_matrix(cm, dir_path, title='Confusion matrix', cmap=plt.cm.Blues, labels=None, normalize=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    混淆矩阵
    参数说明:
        输入:
            cm (narray): 图像数据  行为正确模式, 列为预测模式
            dir_path (string): 保存地址
            title (string):  图像标题
                            Default: 'Confusion matrix'
            cmap : 颜色图实例或注册的颜色图名称
                    Default: plt.cm.Blues
            labels (list): 坐标
                    Default: None
            normalize (bool): 是否归一化
                    Default: False
        输出:

    """
    if labels is None:
        labels = []
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    # cm
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)

    fmt = '.2f' if normalize else '.3f'

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i == j:
            # i,j 代表 cm的行和列数，plt.text 要求输入 坐标(j, i)
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(dir_path + '%s' % title, bbox_inches='tight')


def plot_line_chart(xlabels, ylabels, dir_path, color='red', label=None, title='line chart', xlabel_name='snrs',
                    ylabel_name='acc'):
    """
    绘制折线图
    参数说明:
        输入:
            xlabels (narray): x轴数据
            ylabels (narray): y轴数据
            dir_path (string): 保存地址
            color : 颜色
                    Default: 'red'
            labels (list): 折线的名称
                    Default: None
            title (string):  图像标题
                            Default: 'line chart'

            xlabel_name (string): x轴的标签
                    Default: 'snrs'
            ylabel_name (string): y轴的标签
                    Default: 'acc'
        输出:
    """
    # Plot accuracy curve
    if label is None:
        label = []
    plt.plot(xlabels, ylabels, color=color, label=label)

    plt.xlabel(xlabel_name)
    plt.ylabel(ylabel_name)
    plt.grid(True, linestyle='--', alpha=1)
    plt.legend()  # 显示图例
    plt.title(title)
    plt.savefig(dir_path+title+'.png')
    plt.show()


def print_acc():
    # 打印测试准确率折线图
    filename = './result/lstm/data/lstm_acc.csv'
    csvfile = open(filename, 'r')
    read = csv.reader(csvfile)
    acc = [r for r in read]
    acc = np.array(acc)
    snr = acc[1:, 0].astype(np.int)  # 从string转化为int
    xlabel_name = acc[0, 0]
    test_acc = acc[1:, 1].astype(np.float)
    ylabel_name = acc[0, 1]
    dir_path = './result/lstm/data/'
    label = 'test_acc'
    title = 'lstm classification tet acc'
    plot_line_chart(snr, test_acc, dir_path=dir_path, label=label, title=title, xlabel_name=xlabel_name,
                    ylabel_name=ylabel_name)


def pirnt_loss():
    # 打印训练损失值折线图
    filename = 'D:/Demo/MyJupyter/common_approaches/src_lstm/lstm_loss.csv'
    csvfile = open(filename, 'r')
    read = csv.reader(csvfile)
    loss = [r for r in read]
    loss = np.array(loss)
    epoch = loss[1:, 0].astype(np.int)  # 从string转化为int
    xlabel_name = loss[0, 0]
    train_loss = loss[1:, 1].astype(np.float)
    ylabel_name = loss[0, 1]
    dir_path = './result/lstm/data/'
    label = 'train_loss'
    title = 'lstm classification train loss'
    plot_line_chart(epoch, train_loss, dir_path=dir_path, label=label, title=title, xlabel_name=xlabel_name,
                    ylabel_name=ylabel_name)


if __name__ == '__main__':
    pirnt_loss()




