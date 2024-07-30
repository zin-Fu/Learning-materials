import torch
import numpy as np
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import torch.optim as optim

'''生成数据集'''
num_inputs = 2  # 特征数(输入个数)为2
num_examples = 1000  # 样本数1000
true_w = [2, -3.4]  # 线性回归真实权重
true_b = 4.2  # 偏差
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 特征(就是x)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 标签(true_y)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 加上噪声

'''读取数据：PyTorch提供了data包来读取数据。由于data常用作变量名，我们将导入的data模块用Data代替。在每一次迭代中，我们将随机读取包含10个数据样本的小批量'''
batch_size = 10
dataset = Data.TensorDataset(features, labels)  # 将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)  # 随机读取小批量
'''TensorDataset是一个PyTorch提供的将数据集和标签组合在一起的工具，用于创建一个包含特征张量和标签张量的数据集。
在这个代码中，features是一个形状为(num_examples, num_inputs)的张量，代表有num_examples个样本，每个样本有num_inputs个特征。
labels是形状为(num_examples,)的张量，代表每个样本的标签。将这两个张量作为参数传递给TensorDataset，就会创建一个包含(features, labels)的数据集。

DataLoader是一个PyTorch提供的用于数据读取的工具，它可以从一个数据集中读取数据并生成一个迭代器，每次迭代返回一个小批量的数据。
这里的data_iter就是一个这样的迭代器，它从dataset中读取数据，每次返回batch_size个样本。
通过设置shuffle=True，数据在每个epoch之前会被打乱顺序，从而使模型更好地学习到样本之间的关系。'''


'''定义模型：'''
'''首先，导入torch.nn模块。实际上，“nn”是neural networks（神经网络）的缩写。顾名思义，该模块定义了大量神经网络的层。之前我们已经用过了autograd，而nn就是利用autograd来定义模型。
nn的核心数据结构是Module，它是一个抽象概念，既可以表示神经网络中的某个层（layer），也可以表示一个包含很多层的神经网络。
在实际使用中，最常见的做法是继承nn.Module，撰写自己的网络/层。一个nn.Module实例应该包含一些层以及返回输出的前向传播（forward）方法'''
net = nn.Sequential()  # 定义一个空的神经网络
net.add_module('linear', nn.Linear(num_inputs, 1))  # 加上一层,我们将线性层的名称定义为'linear'，可以使用该名称来访问该层的权重和偏差

# 可以通过net.parameters()来查看模型所有的可学习参数，此函数将返回一个生成器。
# for param in net.parameters():
# print(param)

'''初始化模型参数'''
'''在使用net前，我们需要初始化模型参数，如线性回归模型中的权重和偏差。PyTorch在init模块中提供了多种参数初始化方法。
这里的init是initializer的缩写形式。我们通过init.normal_将权重参数每个元素初始化为随机采样于均值为0、标准差为0.01的正态分布。偏差会初始化为零'''
init.normal_(net[0].weight, mean=0, std=0.01)  # 对第一层权重进行初始化
init.constant_(net[0].bias, val=0)  # 对第一层的偏置向量进行常数初始化

'''定义损失函数'''
loss = nn.MSELoss()

'''定义优化算法'''
optimizer = optim.SGD(net.parameters(), lr=0.03)  # 学习率设为0.03
print(optimizer)

'''训练模型'''
'''在使用Gluon训练模型时，我们通过调用optim实例的step函数来迭代模型参数。按照小批量随机梯度下降的定义，我们在step函数中指明批量大小，从而对批量中样本梯度求平均'''
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:  # 内层循环遍历数据集中的每个样本。其中，X 是特征，y 是标签
        output = net(X)  # 将样本 X 作为输入，通过网络 net 进行前向传播，得到模型的输出结果 output
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad()  # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()  # 优化器根据梯度自动调整模型参数，使得模型在这次迭代中的损失函数值下降
    print('epoch %d, loss: %f' % (epoch, l.item()))  # l.item() 表示损失函数的值（取一个标量）
# 验证
dense = net[0]
print(true_w, dense.weight)
print(true_b, dense.bias)
