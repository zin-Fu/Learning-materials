import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

'''读取和获取数据：使用Fashion-MNIST数据集，并设置批量大小为256'''
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''
初始化模型参数：
使用向量表示每个样本。已知每个样本输入是高和宽均为28像素的图像。模型的输入向量的长度是 28×28=784
28×28=784：该向量的每个元素对应图像中每个像素。由于图像有10个类别，单层神经网络输出层的输出个数为10，因此softmax回归的权重和偏差参数分别为784×10
784×10和1×10
1×10的矩阵
'''
num_inputs = 784  # 设置输入层神经元数量（28*28）
num_outputs = 10  # 设置输出层神经元数量（十个类别）
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)  # 生成随机数，转化成tensor类型
b = torch.zeros(num_outputs, dtype=torch.float)  # 把偏置b初始化为全0向量
W.requires_grad_(requires_grad=True)  # 需要模型参数梯度
b.requires_grad_(requires_grad=True)

'''
实现softmax运算：
在介绍如何定义softmax回归之前，我们先描述一下对如何对多维Tensor按维度操作。
在下面的例子中，给定一个Tensor矩阵X。我们可以只对其中同一列（dim=0）或同一行（dim=1）的元素求和，并在结果中保留行和列这两个维度（keepdim=True）。
'''
# X = torch.tensor([[1, 2, 3], [4, 5, 6]])
# print(X.sum(dim=0, keepdim=True))
# print(X.sum(dim=1, keepdim=True))

'''
下面我们就可以定义前面小节里介绍的softmax运算了。在下面的函数中，矩阵X的行数是样本数，列数是输出个数。
为了表达样本预测各个输出的概率，softmax运算会先通过exp函数对每个元素做指数运算，再对exp矩阵同行元素求和，最后令矩阵每行各元素与该行元素之和相除。
这样一来，最终得到的矩阵每行元素和为1且非负。因此，该矩阵每行都是合法的概率分布。
softmax运算的输出矩阵中的任意一行元素代表了一个样本在各个输出类别上的预测概率。
'''

def softmax(X):
    X_exp = X.exp()  # 将输入张量 X 上的每个元素都取指数，得到一个新的张量 X_exp
    partition = X_exp.sum(dim=1, keepdim=True)  # 对于每行，计算所有元素之和，并保持二维形状以便广播
    return X_exp / partition  # 这里应用了广播机制

X = torch.rand((2, 5))
X_prob = softmax(X)
# print(X_prob, X_prob.sum(dim=1))

'''
定义模型：
有了softmax运算，我们可以定义上节描述的softmax回归模型了。
这里通过view函数将每张原始图像改成长度为num_inputs的向量。
'''
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)
'''首先通过X.view((-1, num_inputs))将X重新变形为(batch_size, num_inputs)的形状，
然后将它与W相乘，再加上偏差b，得到一个(batch_size, num_outputs)的结果。
最后对这个结果使用softmax函数进行处理，并将处理后的结果返回'''

'''
定义损失函数：
上一节中，我们介绍了softmax回归使用的交叉熵损失函数。为了得到标签的预测概率，我们可以使用gather函数。
在下面的例子中，变量y_hat是2个样本在3个类别的预测概率，变量y是这2个样本的标签类别。通过使用gather函数，我们得到了2个样本的标签的预测概率。
与3.4节（softmax回归）数学表述中标签类别离散值从1开始逐一递增不同，在代码中，标签类别的离散值是从0开始逐一递增的。
'''
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])  # 预测概率
y = torch.LongTensor([0, 2])  # 真实标签
y_hat.gather(1, y.view(-1, 1))

def cross_entropy(y_hat, y):  # 交叉熵损失函数
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

'''
计算分类准确率：
给定一个类别的预测概率分布y_hat，我们把预测概率最大的类别作为输出类别。
如果它与真实类别y一致，说明这次预测是正确的。分类准确率即正确预测数量与总预测数量之比。
为了演示准确率的计算，下面定义准确率accuracy函数。其中y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。
相等条件判断式(y_hat.argmax(dim=1) == y)是一个类型为ByteTensor的Tensor，我们用float()将其转换为值为0（相等为假）或1（相等为真）的浮点型Tensor。
'''
def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()  # y_hat.argmax(dim=1)获取预测中最大概率的类别
# print(accuracy(y_hat, y))

'''类似地，我们可以评价模型net在数据集data_iter上的准确率'''
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:  # 遍历数据集中的每个样本，使用net(X)获取模型对该样本的预测结果，然后将预测结果与真实类别进行比较
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()  # 比较结果被累加到acc_sum中
        n += y.shape[0]  # 获取数据集中样本的总数
    return acc_sum / n  # acc_sum被除以n，得到数据集上的平均准确率

# 因为我们随机初始化了模型net，所以这个随机模型的准确率应该接近于类别个数10的倒数即0.1
# print(evaluate_accuracy(test_iter, net))

'''
训练模型：
训练softmax回归的实现跟3.2（线性回归的从零开始实现）一节介绍的线性回归中的实现非常相似。
我们同样使用小批量随机梯度下降来优化模型的损失函数。
在训练模型时，迭代周期数num_epochs和学习率lr都是可以调的超参数。改变它们的值可能会得到分类更准确的模型
'''
num_epochs, lr = 5, 0.1

def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):  # 先遍历整个训练集
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)  # 对于每个小批量数据，将其输入模型得到预测结果 y_hat
            l = loss(y_hat, y).sum()  # 计算损失函数值l

            if optimizer is not None:  # 如果优化器不为空，则将优化器的梯度清零；否则，手动将参数的梯度清零
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # 计算损失函数的梯度，并使用优化器更新模型参数

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  # 累加训练集上的损失和准确率
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)  # 计算测试集上的准确率
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f' % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
'''
epoch 1, loss 0.7909, train acc 0.745, test acc 0.791
epoch 2, loss 0.5713, train acc 0.813, test acc 0.810
epoch 3, loss 0.5256, train acc 0.826, test acc 0.818
epoch 4, loss 0.5011, train acc 0.833, test acc 0.820
epoch 5, loss 0.4852, train acc 0.837, test acc 0.829
'''

'''
预测：
训练完成后，现在就可以演示如何对图像进行分类了。给定一系列图像（第三行图像输出），我们比较一下它们的真实标签（第一行文本输出）和模型预测结果（第二行文本输出）
'''
X, y = next(iter(test_iter))

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])
