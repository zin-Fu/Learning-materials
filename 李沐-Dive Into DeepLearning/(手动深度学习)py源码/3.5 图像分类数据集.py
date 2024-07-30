import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l
import sys
import time

'''
通过torchvision的torchvision.datasets来下载这个数据集
指定参数transform = transforms.ToTensor()使所有数据转换为Tensor
'''
mnist_train = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='~/Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())
'''
这两行代码是用于创建训练集和测试集的数据集对象，这里使用了FashionMNIST数据集，数据集已经在torchvision中预处理好了。
具体来说，这两行代码中的root参数指定了数据集存放的路径，train参数指定了是训练集还是测试集，download参数指定是否需要下载数据集（如果已经下载则可以设为False）
transform参数指定对数据集进行的变换，这里使用transforms.ToTensor()将数据转换为PyTorch中的Tensor类型。
'''

# print(type(mnist_train))
# print(len(mnist_train), len(mnist_test)) # len()来获取该数据集的大小
feature, label = mnist_train[0]  # 可以通过下标来访问任意一个样本
# print(feature.shape, label)  # Channel x Height x Width


def get_fashion_mnist_labels(labels):  # 将数值标签转成相应的文本标签
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):  # 定义一个可以在一行里画出多张图像和对应标签的函数
    d2l.use_svg_display()
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))  # 这里的_表示我们忽略（不使用）的变量
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


'''
现在，我们看一下训练数据集中前10个样本的图像内容和文本标签
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))
'''

'''
我们将在训练数据集上训练模型，并将训练好的模型在测试数据集上评价模型的表现。前面说过，mnist_train是torch.utils.data.Dataset的子类，
所以我们可以将其传入torch.utils.data.DataLoader来创建一个读取小批量数据样本的DataLoader实例。
在实践中，数据读取经常是训练的性能瓶颈，特别当模型较简单或者计算硬件性能较高时。PyTorch的DataLoader中一个很方便的功能是允许使用多进程来加速数据读取。
这里我们通过参数num_workers来设置4个进程读取数据。
'''
batch_size = 256
if sys.platform.startswith('win'):  # 判断是不是windows系统
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)  # 创建训练和测试数据的迭代器
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
'''查看读取一遍训练数据需要的时间'''
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))

