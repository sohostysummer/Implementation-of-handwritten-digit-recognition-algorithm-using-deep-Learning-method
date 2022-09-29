# 基本模块导入
# Basic module import
import torch
import torchvision
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.autograd import Variable

# 调用GPU进行训练
# Using gpu to train
device = torch.device("cuda")

# 数据标准化变换
# 标准差变换法，使用原始数据的均值（Mean）和标准差（Standard Deviation）来进行数据的标准化
# 在进行标准化变换后，数据全部符合均值为0，标准差为1的标准正态分布
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean = [0.5], std = [0.5])])

# 导入训练集数据
# 如果下载速度过慢可以参考修改MNIST路径为本地路径，参考代码如下,修改位置为按住ctrl点击MNIST
# ("file:///XX/Desktop/MNIST_data/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
# ("file:///XX/Desktop/MNIST_data/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
# ("file:///XX/Desktop/MNIST_data/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
# ("file:///XX/Desktop/MNIST_data/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
data_train = datasets.MNIST(root = "./data/",
                            transform = transform,
                            train = True,
                            download = True)
# 导入测试集数据
data_test = datasets.MNIST(root = "./data",
                           transform = transform,
                           train = False)


# dataset参数用于指定我们传入的数据集名称
# batch_size设置了每个包中的图片数据个数，代码中的值为64，所以在每个包中会包含64张图片
# 将shullfe参数设置为True，在装载的时候会将数据随机打乱顺序并进行打包
data_loader_train = torch.utils.data.DataLoader(dataset = data_train,
                                                batch_size = 64,
                                                shuffle = True)

data_loader_test = torch.utils.data.DataLoader(dataset = data_test,
                                               batch_size = 64,
                                               shuffle = True)

# 数据预览
# 使用next和iter操作来获取一个批次的图片数据和其对应的图片标签
images, labels = next(iter(data_loader_train))

# 使用torchvision.utils中的make_grid类方法将一个批次的图片变成网格模式
# 每个批次的装载数据都是4维的，batch_size，channel，height，weight(每一批次中的数据个数，每张图片的色彩通道数，每张图片的高度和宽度)
# 在通过torchvision.utils.make_grid之后图片的维度变成了(channel,height,weight)，批次的图片全部整合到了一起，维度中对应值不一样了，但是色彩通道数保持不变
# 若我们想用Matplotlib将数据显示成正常的图片格式，则使用的数据首先必须是数组，其次这个数组的维度必须是(height,weight,channel)
img = torchvision.utils.make_grid(images)

# 所以我们要通过numpy和transpose完成原始数据类型的转换和数据维度的转换，才可以用Matplotlib绘制出正确的图像
img = img.numpy().transpose(1,2,0)

std = [0.5]

mean = [0.5]

img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)

# 模型搭建和参数优化

# 输入输出相同

class SElayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer,self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(channel, channel // reduction, bias=False),
            torch.nn.ReLU(inplace = True),
            torch.nn.Linear(channel // reduction, channel, bias=False),
            torch.nn.Sigmoid()
            )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Model(torch.nn.Module):

    def __init__(self,n_in,n_out):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Sequential(
            # torch.nn.Conv2d用于搭建卷积神经网络的卷积层
            # 主要的输入参数有输入通道数，输出通道数，卷积核大小，卷积核移动步长，Padding值
            # 其中输入通道数的数据类型是整型，用于确定输入数据的层数
            # 输出通道数的数据类型也是整型，用于确定输出数据的层数
            # torch.nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            # self.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(n_in, n_out, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            torch.nn.ReLU(),
            # torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.Conv2d(n_out, 2*n_out, kernel_size=3, stride=1, padding=1),
            # torch.nn.ReLU(),
            torch.nn.ReLU(),
            # torch.nn.MaxPool2d用于实现卷积神经网络中的最大池化层
            # 主要的输入参数有池化窗口大小，池化窗口移动步长，Padding值
            # torch.nn.MaxPool2d(stride=2,kernel_size=2))
            torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.se = SElayer(2*n_out)

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(14*14*128,1024),
            torch.nn.ReLU(),
            # torch.nn.Dropout类用于防止卷积神经网络在训练过程中发生过拟合
            # 在模型训练过程中，以一定随机概率将卷积神经网络模型的部分参数归零，以达到减少相邻两层神经元神经连接的目的
            # 最后训练出来的模型对各部分的权重参数不产生过度依赖，从而防止过拟合
            # 随机值大小默认为0.5
            torch.nn.Dropout(p = 0.5),
            torch.nn.Linear(1024,10))

    # 前向传播forward函数
    def forward(self,x):
        out = self.conv1(x)
        out = Variable(out)

        out = self.se(out)
        # 对参数实现扁平化
        # 不扁平化会使得全连接层的实际输出的参数维度和其定义输入的维度将不匹配，程序会报错
        out = out.view(-1,14*14*128)
        # 进行最后的分类
        out = self.dense(out)

        return out

#模型的训练和参数优化
n_in = 1
n_out = 64
model = Model(n_in,n_out).to(device)
# 计算损失值的损失函数用的是交叉熵
cost = torch.nn.CrossEntropyLoss()
cost = cost.to(device)
# 优化函数使用的是Adam自适应优化算法，需要优化的参数是在Model中生成的全部参数
# 因为没有定义学习速率的值，所以使用默认值
optimizer = torch.optim.Adam(model.parameters())

# 训练迭代世代
n_epochs = 5

# 定义两个数组，用于后期绘制损失曲线、精确度曲线用
Loss_list = []
Accuracy_list = []

for epoch in range(n_epochs):

    # 初始化损失和准确率
    running_loss = 0.0
    running_correct = 0

    print("Epoch {}/{}".format(epoch, n_epochs))
    # 便于区分五次迭代
    print("_" * 10)

    # 这里设置训练进度条
    for data in tqdm(data_loader_train):
    # for data in tqdm(data_loader_train):

        # 从每个批次中获取数据
        X_train, y_train = data

        X_train = X_train.to(device)
        y_train = y_train.to(device)

        # 张量变变量
        X_train, y_train = Variable(X_train), Variable(y_train)
        # 给模型输入训练数据
        outputs = model(X_train)
        # dim = 1表示输出所在行的最大值
        _, pred = torch.max(outputs.data,1)
        # 清空上一次梯度
        optimizer.zero_grad()
        # 获取损失
        loss = cost(outputs,y_train)
        # 误差反向传播，求梯度
        loss.backward()
        # 优化器参数更新，向梯度方向走了一步
        optimizer.step()
        # 将零维张量换成浮点数
        running_loss += loss.item()
        # 预测值是否与训练值相等
        running_correct += torch.sum(pred == y_train.data)

    testing_correct = 0

    for data in data_loader_test:
        X_test, y_test = data

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        X_test, y_test = Variable(X_test), Variable(y_test)
        outputs = model(X_test)
        _, pred = torch.max(outputs.data, 1)
        testing_correct += torch.sum(pred == y_test.data)

    print("Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)))

    # 在此处保留训练获得的权重
    torch.save(model.state_dict(), './model.pth')
    torch.save(optimizer.state_dict(), './optimizer.pth')

    # Loss_list.append(running_loss / (len(data_train)))
    # Accuracy_list.append(100 * running_correct / (len(data_train)))

    x1 = []
    x2 = []
    y1 = []
    y2 = []
    x1.append(epoch)
    x2.append(epoch)
    # y1 = Accuracy_list
    y1.append(100 * running_correct / (len(data_train)))
    # y2 = Loss_list
    y2.append(running_loss / (len(data_train)))
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    # plt.show()在绘制图形后会暂停程序，需要手动关闭才能继续运行
    plt.draw()
    plt.pause(0.1)
    plt.savefig("accuracy_loss.jpg")