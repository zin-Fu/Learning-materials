import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l
'''获取和读取数据：我们将使用Fashion-MNIST数据集，并设置批量大小为256'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''
定义模型参数：
Fashion-MNIST数据集中图像形状为 28×28
28×28，类别数为10。本节中我们依然使用长度为 28×28=784
28×28=784 的向量表示每一张图像。因此，输入个数为784，输出个数为10。实验中，我们设超参数隐藏单元个数为256
'''
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens)), dtype=torch.float)  # 用随机数初始化第一个全连接层的权重矩阵W1
b1 = torch.zeros(num_hiddens, dtype=torch.float)  # 初始化第一个全连接层的偏置向量b1为0
W2 = torch.tensor(np.random.normal(0, 0.01, (num_hiddens, num_outputs)), dtype=torch.float)  # 同上
b2 = torch.zeros(num_outputs, dtype=torch.float)

params = [W1, b1, W2, b2]  # 将模型参数存储在列表params中
for param in params:
    param.requires_grad_(requires_grad=True)  # 模型参数的requires_grad属性设置为True，表示需要计算梯度

'''定义激活函数：使用基础的max函数来实现ReLU，而非直接调用relu函数'''
def relu(X):
    return torch.max(input=X, other=torch.tensor(0.0))

'''
定义模型：
同softmax回归一样，我们通过view函数将每张原始图像改成长度为num_inputs的向量。
然后我们实现上一节中多层感知机的计算表达式
'''
def net(X):  # 定义神经网络
    X = X.view((-1, num_inputs))
    H = relu(torch.matmul(X, W1) + b1)  # 计算第一个全连接层的输出，输入层到隐藏层的计算
    return torch.matmul(H, W2) + b2  # 隐藏层到输出层的计算

'''定义损失函数：为了得到更好的数值稳定性，我们直接使用PyTorch提供的包括softmax运算和交叉熵损失计算的函数'''
loss = torch.nn.CrossEntropyLoss()

'''
训练模型：
我们直接调用d2lzh_pytorch包中的train_ch3函数，我们在这里设超参数迭代周期数为5，学习率为100.0
'''
num_epochs, lr = 5, 100.0
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

