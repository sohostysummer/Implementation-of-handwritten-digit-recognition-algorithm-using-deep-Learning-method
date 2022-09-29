import torch
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable

transform = transforms.Compose([transforms.ToTensor(),
                                # 数据标准化变换
                                # 标准差变换法，使用原始数据的均值（Mean）和标准差（Standard Deviation）来进行数据的标准化
                                # 在进行标准化变换后，数据全部符合均值为0，标准差为1的标准正态分布
                                transforms.Normalize(mean = [0.5], std = [0.5])])

data_train = datasets.MNIST(root = "./data/",
                            transform = transform,
                            train = True,
                            download = True)

data_test = datasets.MNIST(root = "./data",
                           transform = transform,
                           train = False)



data_loader_test = torch.utils.data.DataLoader(dataset = data_test,
                                               batch_size = 4,
                                               shuffle = True)

X_test, y_test = next(iter(data_loader_test))

inputs = Variable(X_test)

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
model = Model(n_in,n_out)

# demo1 完全加载权重
state_dict = model.state_dict()
weights = torch.load("./model.pth") #读取预训练模型权重
model.load_state_dict(weights)


pred = model(inputs)

_,pred = torch.max(pred, 1)

print("Predict Label is:", [ i for i in pred.data])

print("Real Label is:", [i for i in y_test])

img = torchvision.utils.make_grid(X_test)

img = img.numpy().transpose(1,2,0)

std = [0.5,0.5,0.5]

mean = [0.5,0.5,0.5]

img = img * std + mean

plt.imshow(img)

plt.show()