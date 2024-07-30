import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as d2l
from collections import OrderedDict

'''读取数据：我们仍然使用Fashion-MNIST数据集和上一节中设置的批量大小'''
batch_size = 256  # 设置批量大小
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)  # 加载模型

'''
定义和初始化模型：
softmax回归的输出层是一个全连接层，所以我们用一个线性模块就可以了。因为前面我们数据返回的每个batch样本x的形状为(batch_size, 1, 28, 28), 
所以我们要先用view()将x的形状转换成(batch_size, 784)才送入全连接层
'''
num_inputs = 784  # 模型的输入为28x28的灰度图像，每个像素点取值范围为0-255，经过转换后被展平成一维向量，即784维。
num_outputs = 10  # 输出为一个10维向量，每一维对应数字0-9的概率

class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):  # 定义构造函数
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)  # 创建一个nn.Linear对象

    def forward(self, x):  # x shape: (batch, 1, 28, 28)
        y = self.linear(x.view(x.shape[0], -1))  # 将x作为参数传递给self.linear对象
        return y

net = LinearNet(num_inputs, num_outputs)  # 创建一个实例化对象

class FlattenLayer(nn.Module):  # 对x的形状转换的这个功能自定义一个FlattenLayer，用于将输入的数据展平
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x): # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)  # 将输入张量x按照batch大小进行展平，返回展平后的结果

net = nn.Sequential(  # 定义神经网络(跟38行作用一样)
    # FlattenLayer(),
    # nn.Linear(num_inputs, num_outputs)
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)

init.normal_(net.linear.weight, mean=0, std=0.01)  # 用均值为0、标准差为0.01的正态分布随机初始化模型的权重参数
init.constant_(net.linear.bias, val=0)

'''定义损失函数：PyTorch提供了一个包括softmax运算和交叉熵损失计算的函数'''
loss = nn.CrossEntropyLoss()

'''定义优化算法：使用学习率为0.1的小批量随机梯度下降作为优化算法'''
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

'''训练模型'''
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
