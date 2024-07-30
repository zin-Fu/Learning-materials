import matplotlib
import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d2l

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05

features = torch.randn((n_train + n_test, num_inputs))
labels = torch.matmul(features, true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]

'''初始化模型参数：首先，定义随机初始化模型参数的函数。该函数为每个参数都附上梯度'''
def init_params():
    w = torch.randn((num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

'''定义l2范数惩罚项'''
def l2_penalty(w):
    return (w**2).sum() / 2

'''定义训练和测试'''
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = d2l.linreg, d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

def fit_and_plot(lambd):  # 其中 lambd 是一个参数，用于控制 L2 范数的惩罚力度
    w, b = init_params()  # 返回参数 w 和 b 的初始化值
    train_ls, test_ls = [], []  # 创建了空列表 train_ls 和 test_ls 用于存储训练和测试集的损失函数值
    for _ in range(num_epochs):
        for X, y in train_iter:
            # 添加了L2范数惩罚项
            l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l = l.sum()  # 通过调用l.sum()将其所有元素求和为一个标量值。这是为了得到一个平均的损失值，用于监测训练进程
            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()
            l.backward()
            d2l.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b), train_labels).mean().item())
        test_ls.append(loss(net(test_features, w, b), test_labels).mean().item())  # 计算训练集和测试集上的损失值，并将其添加到相应的列表train_ls和test_ls中
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])  # 用两个集合里的点来画图
    print('L2 norm of w:', w.norm().item())

'''观察过拟合'''
fit_and_plot(lambd=0)

'''使用权重衰减'''
fit_and_plot(lambd=3)

#output
# L2 norm of w: 12.395933151245117
# L2 norm of w: 0.03533049672842026
