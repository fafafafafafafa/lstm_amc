import dataset
import numpy as np
import data_deal
import model
import torch
import torch.nn
import torch.utils.data
import pytorchtools
import csv
"""
命名规则:
函数: 小写 分隔符_
类: 驼峰准则
变量: 小写 分隔符_
"""

# 把剩下的代码整合完,确认能跑通,整理一些常用的语句
# 再有时间看看lstm源代码


def train(batch_size, my_model, loader_train, criterion, optimizer):
    """
    单次epoch训练
    参数说明:
        输入:
            my_model: 训练的网络模型
            loader_train: 数据加载器
            criterion: 损失函数
            optimizer: 优化器
        输出:
            train_loss: 单次训练的误差
    """
    my_model.train()
    # 预测的正确数
    running_correct = 0
    # 损失值
    running_loss = 0.0
    train_loss = 0
    for batch, (signals, labels) in enumerate(loader_train, 1):

        signals = torch.transpose(signals, 2, 1)
        if torch.cuda.is_available():
            # 获取输入数据X和标签Y并拷贝到GPU上
            # 注意有许多教程再这里使用Variable类来包裹数据以达到自动求梯度的目的，如下
            # Variable(imgs)
            # 但是再pytorch4.0之后已经不推荐使用Variable类，Variable和tensor融合到了一起
            # 因此我们这里不需要用Variable
            # 若我们的某个tensor变量需要求梯度，可以用将其属性requires_grad=True,默认值为False
            # 如，若X和y需要求梯度可设置X.requires_grad=True，y.requires_grad=True
            # 但这里我们的X和y不需要进行更新，因此也不用求梯度

            X, y = signals.cuda(), labels.cuda()

        else:
            X, y = signals, labels

        # 将输入X送入模型进行训练
        outputs = my_model(X)  # torch.Size([batch_size, len(mods)])
        # torch.max()返回两个字，其一是最大值，其二是最大值对应的索引值
        # 这里我们用y_pred接收索引值
        temp, y_pred = torch.max(outputs.detach(), dim=1)  # torch.Size([batch_size, len(mods)])

        # 在求梯度前将之前累计的梯度清零，以免影响结果
        optimizer.zero_grad()
        # 计算损失值
        # 注意 outputs,y 维度相同
        loss = criterion(outputs, y)
        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 计算一个批次的损失值和
        running_loss += loss.detach().item()
        # 计算一个批次的预测正确数
        temp, y = torch.max(y.detach(), dim=1)
        running_correct += torch.sum(y_pred == y)

        # 打印训练结果
        if batch == len(loader_train):
            train_loss = running_loss / batch
            acc = 100 * running_correct.item() / (batch_size * batch)
            print(
                'Batch {batch}/{iter_times},Train Loss:{loss:.4f},Train Acc:{correct}/{lens}={acc:.4f}%'.format(
                    batch=batch,
                    iter_times=len(loader_train),
                    loss=running_loss / batch,
                    correct=running_correct.item(),
                    lens=batch_size * batch,
                    acc=acc
                ))
    return train_loss


def main():
    filename = "../../src/RML2016.10a_dict.pkl"
    X_train, X_test, Y_train, Y_test = dataset.get_dataRML2016a(filename, 2016)
    print("X_train", X_train.shape)
    print("Y_train", Y_train.shape)
    X_train, X_test = data_deal.to_amp_phase(X_train, X_test)
    print("X_train", X_train.shape)
    # 参数设置
    batch_size = 200
    epochs = 150
    lr = 1e-3  # 学习率

    data_loss = []  # 记录训练data_loss[epoch, train_loss]
    # 加载数据
    dst = dataset.SigDataset(X_train, Y_train)
    loader_train = torch.utils.data.DataLoader(dst, batch_size=batch_size, shuffle=True)
    # 构建模型
    my_model = model.Lstm2(2, 128, 11, 2)
    print(my_model.parameters())
    # 定义损失函数
    criterion = torch.nn.CrossEntropyLoss()
    # 定义优化器
    optimizer = torch.optim.Adam(my_model.parameters(), lr=lr, betas=(0.9, 0.999))
    # 将模型的所有参数拷贝到到GPU上
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        my_model = my_model.cuda()
    # 初始化 earlying stopping 对象
    patience = 10
    delta = 0.01
    dir_path = './result/lstm/checkpoints/'
    model_name = 'lstm.pt'
    early_stopping = pytorchtools.EarlyStopping(patience=patience, verbose=True, delta=delta, dir_path=dir_path,
                                                filename=model_name)
    # for epoch in tqdm.tqdm(range(1, epoches+1)):
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        train_loss = train(batch_size, my_model, loader_train, criterion, optimizer)
        data_loss.append([epoch, train_loss])
        early_stopping(train_loss, my_model)
        if early_stopping.early_stop:
            print("early_stop")
            break
    # 保存数据
    dir_path = './result/lstm/data/'
    filename = 'lstm_loss.csv'
    # 将loss写入csv
    csvfile = open(dir_path+filename, "w", newline="")  # w覆盖， a追加
    writer = csv.writer(csvfile)
    writer.writerow(['epoch', 'train_loss'])
    writer.writerows(data_loss)
    csvfile.close()


if __name__ == '__main__':
    main()
