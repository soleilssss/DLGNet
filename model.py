"""resnet in pytorch

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn
from IGloss import IGLoss
from cbam import CBAM
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=2, inputchannel=1):
        super().__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        # self.cbam2_x = CBAM(64)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        # self.cbam3_x = CBAM(128)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        # self.cbam4_x = CBAM(256)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.cbam5_x = CBAM(512)

        self.in_channels = 64
        self.conv11 = nn.Conv2d(in_channels=inputchannel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn11 = nn.BatchNorm2d(num_features=64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_x1 = self._make_layer(block, 64, num_block[0], 1)
        # self.cbam2_x1 = CBAM(64)
        self.conv3_x1 = self._make_layer(block, 128, num_block[1], 2)
        # self.cbam3_x1 = CBAM(128)
        self.conv4_x1 = self._make_layer(block, 256, num_block[2], 2)
        # self.cbam4_x1 = CBAM(256)
        self.conv5_x1 = self._make_layer(block, 512, num_block[3], 2)
        self.cbam5_x1 = CBAM(512)


        self.avg_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = IGLoss(512 * block.expansion, num_classes)
        self.fc2 = IGLoss(512 * block.expansion, num_classes)
        self.fc  = IGLoss(512* 2 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, img1, img2, target = None):
        output1 = self.conv1(img1)
        output1 = self.bn1(output1)
        output1 = self.relu(output1)
        output1 = self.maxpool(output1)
        output1 = self.conv2_x(output1)
        # output1 = self.cbam2_x(output1)
        output1 = self.conv3_x(output1)
        # output1 = self.cbam3_x(output1)
        
        output2 = self.conv11(img2)
        output2 = self.bn11(output2)
        output2 = self.relu1(output2)
        output2 = self.maxpool1(output2)
        output2 = self.conv2_x1(output2)
        # output2 = self.cbam2_x1(output2)
        output2 = self.conv3_x1(output2)
        # output2 = self.cbam3_x1(output2)
        
        
        output1 = self.conv4_x(output1)
        # output1 = self.cbam4_x(output1)
        output1 = self.conv5_x(output1)
        output1 = self.cbam5_x(output1)

        output2 = self.conv4_x1(output2)
        # output2 = self.cbam4_x1(output2)
        output2 = self.conv5_x1(output2)
        output2 = self.cbam5_x1(output2)

        # 同时输出3个预测，根据三个预测进行优化
        c1 = self.avg_pool1(output1)
        c1 = c1.view(c1.size(0),-1)
        c1, _, _ = self.fc1(c1, labels = target)
        c2 = self.avg_pool2(output2)
        c2 = c2.view(c2.size(0),-1)
        c2, _, _ = self.fc2(c2, labels = target)

        
        output = torch.cat((output1, output2),dim = 1)
        output = self.avg_pool(output)
        exoutput = output.view(output.size(0), -1)
        output, _, _ = self.fc(exoutput, labels = target)
        return exoutput, c1, c2, output


def resnet18(num_classes=4, inputchannel=3):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=4, inputchannel=3)

def resnet34(num_classes=4, inputchannel=3):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3],num_classes=4, inputchannel=3)

def resnet50(num_classes=4, inputchannel=3):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3],num_classes=4, inputchannel=3)

def resnet101(num_classes=4, inputchannel=3):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=4, inputchannel=3)

def resnet152(num_classes=4, inputchannel=3):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=4, inputchannel=3)



