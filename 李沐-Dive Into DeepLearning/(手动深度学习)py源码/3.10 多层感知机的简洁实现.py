import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l

'''
定义模型
和softmax回归唯一的不同在于，我们多加了一个全连接层作为隐藏层。
它的隐藏单元个数为256，并使用ReLU函数作为激活函数
'''
num_inputs, num_outputs, num_hiddens = 784, 10, 256

net = nn.Sequential(
        d2l.FlattenLayer(),  # 将输入数据展平，将每个样本变成一个向量
        nn.Linear(num_inputs, num_hiddens),  # 一个全连接层，它将输入数据的每个特征连接到隐藏层的每个神经元
        nn.ReLU(),  # 使用ReLU函数作为激活函数，对隐藏层的输出进行非线性变换
        nn.Linear(num_hiddens, num_outputs),  # 一个全连接层，它将隐藏层的每个神经元连接到输出层的每个神经元
        )

for params in net.parameters():  # 对神经网络中的参数进行初始化
    init.normal_(params, mean=0, std=0.01)

'''读取数据并训练模型'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
