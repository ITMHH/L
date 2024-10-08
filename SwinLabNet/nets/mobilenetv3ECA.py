import math
import os
#V3主干+eca注意力机制
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

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

class ECA(nn.Module):
    def __init__(self, input_channel, gamma=2, b=1):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Ensure kernel size is not greater than input_channel
        kernel_size = min(3, max(1, input_channel // gamma))
        self.conv = nn.Conv1d(input_channel, input_channel, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.b = b

    def forward(self, x):
        batch, channel, _, _ = x.size()

        y = self.avg_pool(x)
        y = self.conv(y.view(batch, channel, -1))
        y = self.sigmoid(y)

        return x * (self.b + y.view(batch, channel, 1, 1))

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, gamma=2, b=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            BatchNorm2d(hidden_dim),
            nn.Hardswish(inplace=True),
            ECA(hidden_dim, gamma, b),  # Include ECA module here
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
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            [1, 16, 1, 1, 1],
            [4, 24, 2, 2, 0],
            [3, 40, 2, 2, 1],
            [3, 80, 3, 2, 0],
            [6, 112, 3, 1, 1],
            [6, 160, 3, 1, 1],
            [6, 320, 1, 1, 0],
        ]

        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(3, input_channel, 2)]

        for t, c, n, s, ECA_flag in interverted_residual_setting:
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
