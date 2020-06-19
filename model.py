import torch
import torch.nn as nn
import torch.nn.functional as F
from utilities.disout import Disout,LinearScheduler


def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)



class DD_CNN(nn.Module):
    def __init__(self):
        super(DD_CNN, self).__init__()
        dist_prob = 0.20
        alpha = 1.0
        block_size = 4
        nr_steps = 57600
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), groups=64)
        self.disout1 = LinearScheduler(Disout(dist_prob=dist_prob, block_size=block_size, alpha=alpha),
                                       start_value=0., stop_value=dist_prob, nr_steps=nr_steps)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), groups=32)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), groups=32)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=(5, 5), stride=(1, 1),
                               padding=(2, 2), groups=8)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc = nn.Linear(256, 3)

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.conv4)
        init_layer(self.conv5)

        init_layer(self.fc)

        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)

    def forward(self, input):
        x = input

        x = self.bn1(self.conv1(x))

        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = F.relu_(self.bn5(self.conv5(x)))
        x = F.max_pool2d(x, kernel_size=(2, 2))

        x = self.disout1(x)

        x = torch.mean(x, dim=3)  # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)  # (batch_size, feature_maps)
        x = self.fc(x)
        output = F.log_softmax(x, dim=-1)

        return output