import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        nn.init.normal_(self.conv.weight.data, 0.0, 0.02)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)


        return [h_next, c_next]

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class DenseResidualBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(DenseResidualBlock, self).__init__()
        self.res_scale = res_scale

        self.b1 = nn.Sequential(*[nn.Conv2d(1 * filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU()])
        self.b2 = nn.Sequential(*[nn.Conv2d(2 * filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU()])
        self.b3 = nn.Sequential(*[nn.Conv2d(3 * filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU()])
        self.b4 = nn.Sequential(*[nn.Conv2d(4 * filters, filters, 3, 1, 1, bias=True), nn.LeakyReLU()])
        self.b5 = nn.Sequential(*[nn.Conv2d(5 * filters, filters, 3, 1, 1, bias=True)])

    def forward(self, x):
        inputs = x

        out = self.b1(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b2(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b3(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b4(inputs)
        inputs = torch.cat([inputs, out], 1)
        out = self.b5(inputs)
        return out.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, filters, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.res_scale = res_scale
        self.dense_blocks = nn.Sequential(
            DenseResidualBlock(filters), DenseResidualBlock(filters), DenseResidualBlock(filters)
        )

    def forward(self, x):
        return self.dense_blocks(x).mul(self.res_scale) + x

class GeneratorRRDB(nn.Module):
    def __init__(self, channels, filters=32, num_res_blocks=2):
        super(GeneratorRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(2*channels, filters, kernel_size=3, stride=1, padding=1)
        # Residual blocks
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks)])
        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        return out


class GeneratorLSTMRRDB(nn.Module):
    def __init__(self, channels, filters=32, num_res_blocks=2):
        super(GeneratorLSTMRRDB, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(2 * channels, filters, kernel_size=3, stride=1, padding=1)

        # Residual blocks
        res_block_list1 = [ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks//2)]
        self.res_blocks1 = nn.Sequential(*res_block_list1)

        self.convlstm = ConvLSTMCell(input_dim = filters, hidden_dim = filters, kernel_size = (3,3), bias = False)

        res_block_list2 = [ResidualInResidualDenseBlock(filters) for _ in range(num_res_blocks // 2)]
        self.res_blocks2 = nn.Sequential(*res_block_list2)

        # Second conv layer post residual blocks
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        # Final output block
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x, hidden_state = None):
        b, _, h, w = x.size()
        if hidden_state is None:
            hidden_state = self.convlstm.init_hidden(batch_size=b, image_size=(h, w))
        out1 = self.conv1(x)
        out = self.res_blocks1(out1)
        out, c_ = self.convlstm(input_tensor=out, cur_state=hidden_state)
        hidden_state = [out, c_]
        out = self.res_blocks2(out)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.conv3(out)
        return out, hidden_state


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)