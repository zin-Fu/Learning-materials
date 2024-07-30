import matplotlib
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
'''构造一个简单的人工训练数据集'''
num_inputs = 2  # 特征数(输入个数)为2
num_examples = 1000  # 样本数1000
true_w = [2, -3.4]  # 线性回归真实权重
true_b = 4.2  # 偏差
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)  # 特征(就是x)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # 标签(true_y)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)  # 加上噪声
'''
在深度学习中，添加噪声有时被用于正则化模型，可以帮助防止过拟合，提高模型的泛化能力。
在这个例子中，噪声被添加到标签中，这可以帮助模型更好地学习数据的分布，防止过度拟合。
添加的噪声是从正态分布中随机采样得到的，其均值为0，标准差为0.01。
这个噪声的大小足够小，不会严重改变标签的分布，但足以产生一定的噪声，帮助模型更好地学习数据的特征。
'''

'''读取数据：在训练模型的时候，我们需要遍历数据集并不断读取小批量数据样本。这里我们定义一个函数：它每次返回batch_size（批量大小）个随机样本的特征和标签'''
def data_iter(batch_size, features, labels):
    num_examples = len(features)  # 获取数据集的样本数
    indices = list(range(num_examples))  # 生成所有样本的索引列表
    random.shuffle(indices)  # 将所有样本的索引随机打乱，实现样本的随机读取
    for i in range(0, num_examples, batch_size):  # 循环读取所有小批量样本
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)  # 函数根据随机索引选取小批量的特征和标签，并通过 yield 关键字返回这些数据


'''将权重初始化成均值为0、标准差为0.01的正态随机数，偏差则初始化成0。'''
batch_size = 10
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32, requires_grad=True)
b = torch.zeros(1, dtype=torch.float32, requires_grad=True)  # requires_grad=True别漏

'''线性回归的矢量计算表达式的实现，使用mm函数做矩阵乘法'''
def linreg(X, w, b):
    return torch.mm(X, w) + b


'''定义线性回归的损失函数'''
def squared_loss(y_hat, y):
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


'''以下的sgd函数实现了上一节中介绍的小批量随机梯度下降算法。它通过不断迭代模型参数来优化损失函数。这里自动求梯度模块计算得来的梯度是一个批量样本的梯度和。我们将它除以批量大小来得到平均值'''
def sgd(params, lr, batch_size):  # 小批量随机梯度下降算法
    for param in params:
        param.data -= lr * param.grad / batch_size  # 这个公式可以理解为每次更新参数时，将参数沿着负梯度方向移动一步，步长为 lr * param.grad / batch_size
        param.grad.data.zero_()  # 清零梯度
        ''' param.data 是一个张量(tensor)，它包含了该参数的实际数值。在 PyTorch 中，模型的参数被存储在一个叫做 Parameter 的类中，Parameter 对象其实是 Tensor 的子类，
        它除了拥有 Tensor 的所有属性和方法之外，还有一个额外的属性 grad，用于存储该参数的梯度值。
        在训练过程中，我们需要对模型参数进行更新。而直接修改 Parameter 对象是不安全的，因为它可能会打破计算图的连续性。
        为了安全地更新模型参数，我们可以通过修改 param.data 属性来更新该参数的实际数值，而不会破坏计算图的连续性'''

lr = 0.03  # 学习率
num_epochs = 3  # 迭代周期个数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除. X和y分别是小批量样本的特征和标签）
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

        # 不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

print(true_w, '\n', w)
print(true_b, '\n', b)
#output
epoch 1, loss 0.035728
epoch 2, loss 0.000133
epoch 3, loss 0.000049
[2, -3.4] 
 tensor([[ 1.9999],
        [-3.3998]], requires_grad=True)
4.2 
 tensor([4.1997], requires_grad=True)
