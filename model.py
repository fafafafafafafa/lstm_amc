import torch
import torch.nn as nn
from torchinfo import summary


"""
放置模型网络
版本说明:
V1
时间: 2022.10.21
作者: fff
说明: lstm1, lstm2
"""


class Lstm1(nn.Module):
    """
    双路rnn，分别处理 两路信号
    参数说明:
        以(batch_size, ndots=128, channels=2, 1) 为例
        意味着 timesteps = ndots, 每个timesteps 输入特征维度为1
        输入:
            input_size: 1
            hidden_size: 隐藏层特征数
            output_size: 分类时输出数
            num_layer: lstm层数
        输出:
            size:(batch_size, output_size)  type: tensor

    """
    def __init__(self, input_size=1, hidden_size=128, output_size=11, num_layer=2):
        super(Lstm1, self).__init__()
        # 定义LSTM
        self.rnn1 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, dropout=0.5)
        self.rnn2 = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, dropout=0.5)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为11
        self.reg = nn.Sequential(
            nn.Linear(2 * hidden_size, output_size)
        )

    def forward(self, x):
        # print("x.shape:",x.shape)  # torch.Size([batch_size, ndots, channels=2, 1])
        x1 = x[:, :, 0, :]
        x2 = x[:, :, 1, :]
        # print("-x1.shape:",x1.shape)  #  torch.Size([batch_size, ndots, 1])
        x1, (ht, ct) = self.rnn1(x1)
        x2, (ht, ct) = self.rnn2(x2)
        # seq_len, batch_size, hidden_size= x1.shape
        # print("--x1.shape:",x1.shape)  # torch.Size([batch_size, ndots, hidden_size])

        x1 = x1[:, -1, :]
        x2 = x2[:, -1, :]
        # print("----x1.shape:",x1.shape)  # torch.Size([batch_size, hidden_size])
        x = torch.cat([x1, x2], -1)
        # print("----x1----.shape:",x.shape)   # torch.Size([batch_size, 2*hidden_size])
        x = self.reg(x)
        # print("----x----.shape:",x.shape)  # torch.Size([batch_size, output_size])
        return x


class Lstm2(nn.Module):
    """
    单路rnn，同时两路信号
    参数说明:
        以(batch_size, ndots=128, channels=2) 为例
        意味着 timesteps = ndots, 每个timesteps 输入特征维度为2,即同一时刻IQ或AP信号
        输入:
            input_size: 2
            hidden_size: 隐藏层特征数
            output_size: 分类时输出数
            num_layer: lstm层数
        输出:
            size:(batch_size, output_size)  type: tensor

    """
    def __init__(self, input_size=2, hidden_size=128, output_size=11, num_layer=2):
        super(Lstm2, self).__init__()
        # 定义LSTM
        self.rnn = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True, dropout=0.5)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为11
        self.reg = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # print("----x.shape:",x.shape)  torch.Size([batch_size, ndots, channels=2])
        x, (ht, ct) = self.rnn(x)
        # print("----x.shape:",x.shape)  # torch.Size([batch_size, ndots, hidden_size])

        x = x[:, -1, :]
        # print("----x.shape:",x.shape)  # torch.Size([batch_size, hidden_size])
        x = self.reg(x)
        # print("----x.shape:",x.shape)   torch.Size([batch_size, output_size])
        return x


if __name__ == "main":
    my_model = Lstm1(1, 128, 11, 2)
    if torch.cuda.is_available():
        my_model = my_model.cuda()

    summary(my_model, (400, 128, 2, 1))


