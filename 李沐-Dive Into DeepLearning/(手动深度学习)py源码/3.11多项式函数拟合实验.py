import torch
import numpy as np
import sys
import matplotlib
import d2lzh_pytorch as d2l

'''生成数据集'''
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5  # 训练数据集大小为100，测试数据集大小为100，真实的模型参数是[1.2, -3.4, 5.6]和5
features = torch.randn((n_train + n_test, 1))  # 生成一个大小为(n_train + n_test, 1)的特征矩阵，每个元素都是从标准正态分布中随机采样得到的。
poly_features = torch.cat((features, torch.pow(features, 2), torch.pow(features, 3)), 1)  # 生成一个大小为(n_train + n_test, 3)的多项式特征矩阵，每个元素都是从features中得到的元素的平方和立方
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]  # 用真实的模型参数和多项式特征矩阵计算标签
          + true_w[2] * poly_features[:, 2] + true_b)
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
'''
看一看生成的数据集的前两个样本
var = features[:2], poly_features[:2], labels[:2]
print(var)
'''

'''定义，训练和测试模型'''
'''我们先定义作图函数semilogy，其中y轴使用了对数尺度'''
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
'''
和线性回归一样，多项式函数拟合也使用平方损失函数。
因为我们将尝试使用不同复杂度的模型来拟合生成的数据集，所以我们把模型定义部分放在fit_and_plot函数中。
多项式函数拟合的训练和测试步骤与3.6节（softmax回归的从零开始实现）介绍的softmax回归中的相关步骤类似
'''
num_epochs, loss = 100, torch.nn.MSELoss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):  # 这个函数接受训练和测试数据集的特征和标签，并返回一个训练好的神经网络模型
    net = torch.nn.Linear(train_features.shape[-1], 1)  # 定义了一个包含一个线性层的神经网络模型，它的输入维度是训练特征的最后一个维度，输出维度为1
    # 通过Linear文档可知，pytorch已经将参数初始化了，所以我们这里就不手动初始化了

    batch_size = min(10, train_labels.shape[0])  # 创建了一个大小为batch_size的随机训练数据迭代器，用于将训练数据划分为小批量，以便进行随机梯度下降
    dataset = torch.utils.data.TensorDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 优化器
    train_ls, test_ls = [], []
    for _ in range(num_epochs):  # num_epochs次循环，每次循环通过训练数据的小批量对神经网络模型进行更新，并记录训练和测试损失
        for X, y in train_iter:
            l = loss(net(X), y.view(-1, 1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()  # 更新模型参数，即执行一步梯度下降更新参数。
        train_labels = train_labels.view(-1, 1)
        test_labels = test_labels.view(-1, 1)  # 将训练标签和测试标签转换为列向量
        train_ls.append(loss(net(train_features), train_labels).item())
        test_ls.append(loss(net(test_features), test_labels).item())  # 分别计算训练集和测试集的损失，并将损失值加入到对应的列表train_ls和test_ls中
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net.weight.data,
          '\nbias:', net.bias.data)

'''三阶多项式函数拟合'''
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
            labels[:n_train], labels[n_train:])

'''线性函数拟合(欠拟合)'''
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])

'''训练样本不足(过拟合)'''
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])

