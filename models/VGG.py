import torch.nn as nn
import torch

# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, class_num=1000, init_weights=False):
        super(VGG, self).__init__()
        #   初始化特征提取网络
        self.features = features
        #   初始化全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512 * 7 * 7, 2048),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, class_num)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#   特征提取网络构造函数 根据传入对应配置的cfg列表，构造对应的提取网络
def make_features(cfg:list):
    #   定义空列表，用来存放创建的每一层结构
    layers = []
    #   初始化输入通道数(RGB图像)
    in_channels = 3
    #   遍历配置列表
    for v in cfg:
        #   如果当前配置元素是"M"字符
        if v == "M":
            #   创建最大池化下采样层
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        #   否则
        else:
            #   创建卷积操作
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            #   将定义好的卷积层与激活函数拼接在一起，添加到layers列表中
            layers += [conv2d, nn.ReLU(True)]
            #   更新通道数
            in_channels = v
    #   通过非关键字参数*layers，将特征提取网络输入nn.Sequential()进行整合
    return nn.Sequential(*layers)

#   定义模型配置的字典文件
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg(model_name="vgg16", **kwargs):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    #   实例化模型  **kwargs 代表可变长度的字典变量
    model = VGG(make_features(cfg), **kwargs)
    return model
