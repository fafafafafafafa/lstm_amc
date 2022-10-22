import numpy as np
import torch
import torch.utils.data
import model
import dataset
import data_deal
import scipy.io
import matplotlib.pyplot as plt
import utils
import csv


def test():
    filename = "../../src/RML2016.10a_dict.pkl"
    mods, snrs, X, lbls = dataset.get_keys_valuesRML2016a(filename)
    seed = 2016
    onehot_flag = True
    X_train, X_test, Y_train, Y_test, test_idx = dataset.split_RML2016a(seed, X, lbls, mods, onehot_flag)
    print("X_test", X_test.shape)
    print("Y_test", Y_test.shape)
    X_train, X_test = data_deal.to_amp_phase(X_train, X_test)
    print("X_test", X_test.shape)
    # 参数设置
    batch_size = 200

    # 加载训练好的模型
    model_file = './result/lstm/checkpoints/lstm.pt'
    my_model = model.Lstm2(input_size=2, hidden_size=128, output_size=11, num_layer=2)
    my_model.load_state_dict(torch.load(model_file))
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    # 测试模式
    my_model.eval()
    # 记录精度
    acc = []
    for snr in snrs:

        # test_SNRs = list(map(lambda x: lbls[x][1], test_idx))
        test_SNRs = list(map(lambda x: lbls[x][1], test_idx))
        test_X_i = X_test[np.where(np.array(test_SNRs) == snr)[0]]
        test_Y_i = np.array(Y_test)[np.where(np.array(test_SNRs) == snr)[0]]

        dst3 = dataset.SigDataset(test_X_i, test_Y_i)
        loader_test = torch.utils.data.DataLoader(dst3, batch_size=batch_size, shuffle=False)
        # 混淆矩阵
        conf = np.zeros([len(mods), len(mods)])
        confnorm = np.zeros([len(mods), len(mods)])  # conf 归一化
        for batch, (signals, labels) in enumerate(loader_test, 1):
            signals = torch.transpose(signals, 2, 1)
            if torch.cuda.is_available():
                X, y = signals.cuda(), labels.cuda()
            else:
                X, y = signals, labels  # y.size() torch.Size([128])

            outputs = my_model(X)  # outputs.size() torch.Size([128, 11])
            _, pred = torch.max(outputs.detach(), dim=1)  # pred.size() torch.Size([128])
            _, y = torch.max(y.detach(), dim=1)  # pred.size() torch.Size([128])

            for i in range(0, pred.shape[0]):
                # 行为正确模式， 列为预测模式
                j = y[i]
                k = pred[i]
                conf[j, k] = conf[j, k] + 1

        # 归一化
        for i in range(0, len(mods)):
            confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
        # 记录数据, 数据整体准确率折线图， 不同信噪比下的混淆矩阵
        dir_path = './result/lstm/data/'
        mat_name = 'snr=%s.mat' % snr
        scipy.io.savemat(dir_path+mat_name, {'confnorm': confnorm})

        plt.figure()
        utils.plot_confusion_matrix(confnorm, dir_path, labels=mods, title="Lstm Confusion Matrix (SNR=%d)" % snr)

        cor = np.sum(np.diag(conf))
        ncor = np.sum(conf) - cor
        print("snr:%s, Overall Accuracy: " % snr, cor / (cor + ncor))

        acc.append([snr, 1.0 * cor / (cor + ncor)])
        acc_name = 'lstm_acc'
        filename = dir_path + acc_name + ".csv"
        # 将acc写入csv
        csvfile = open(filename, "w", newline="")  # w覆盖， a追加
        writer = csv.writer(csvfile)
        writer.writerow(['snr', 'test_acc'])
        writer.writerows(acc)
        csvfile.close()


if __name__ == '__main__':
    print("开始测试-----------")
    test()
