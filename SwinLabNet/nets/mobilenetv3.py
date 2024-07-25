import math
import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
##########################训练成功得v3主干网络+se注意力机制
BatchNorm2d = nn.BatchNorm2d

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        BatchNorm2d(oup),
        nn.Hardswish(inplace=True)  # MobileNetV3 uses h-swish activation
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        BatchNorm2d(oup),
        nn.Hardswish(inplace=True)  # MobileNetV3 uses h-swish activation
    )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channel, squeeze_factor=4):
        super(SqueezeExcitation, self).__init__()
        squeeze_channels = input_channel // squeeze_factor
        self.fc1 = nn.Conv2d(input_channel, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channel, kernel_size=1)
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        out = torch.mean(x, dim=(-2, -1), keepdim=True)
        out = self.fc1(out)
        out = nn.ReLU(inplace=True)(out)
        out = self.fc2(out)
        out = self.activation(out)
        return x * out
#实现到残差结构
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
#计算隐藏层的通道数，等于输入通道数乘以扩展比例并四舍五入取整。判断是否使用残差连接，当步长为1且输入通道数等于输出通道数时使用残差连接。
        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
#定义一个卷积层序列
        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),#1x1卷积层，不使用偏置项。
            BatchNorm2d(hidden_dim),#批量归一化层。
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),#激活函数Hardswish。
            SqueezeExcitation(hidden_dim),  # Add SE module  SE模块，用于通道注意力机制。
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV3(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV3, self).__init__()
        block = InvertedResidual
        input_channel = 16  # According to MobileNetV3 implementation
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s, SE
            [1, 16, 1, 1, 1],  # 224, 224, 3 -> 112, 112, 16
            [4, 24, 2, 2, 0],  # 112, 112, 16 -> 56, 56, 24
            [3, 40, 2, 2, 1],  # 56, 56, 24 -> 28, 28, 40
            [3, 80, 3, 2, 0],  # 28, 28, 40 -> 14, 14, 80
            [6, 112, 3, 1, 1],  # 14, 14, 80 -> 14, 14, 112
            [6, 160, 3, 1, 1],  # 14, 14, 112 -> 14, 14, 160
            [6, 320, 1, 1, 0],  # 14, 14, 160 -> 7, 7, 320
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # 512, 512, 3 -> 224, 224, 16
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s, SE in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url, model_dir=model_dir)

def mobilenetv3(pretrained=False, **kwargs):
    model = MobileNetV3(n_class=1000, **kwargs)
    if pretrained:
        model.load_state_dict(load_url('URL_TO_YOUR_MOBILENETV3_MODEL_WEIGHTS.pth.tar'), strict=False)
    return model

if __name__ == "__main__":
    model = mobilenetv3()
    for i, layer in enumerate(model.features):
        print(i, layer)
