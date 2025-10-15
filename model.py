import torch
from torch import nn


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('Initialization method [{}] is not implemented'.format(init_type))
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)

    print("=== Initialize network with [{}] ===".format(init_type))
    net.apply(init_func)


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners,
                          recompute_scale_factor=True)
        return out


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class spatial_attn_layer(nn.Module):
    def __init__(self, channel, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = nn.Sequential(nn.ReflectionPad2d(int(kernel_size // 2)),
                                     nn.Conv2d(2, 1, kernel_size, 1, 0, 1, 1, True))

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


## ------ Channel Attention --------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=False):
        super(ca_layer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##---------- Dual Attention Unit (DAU) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, bn=False, act=nn.LeakyReLU(), res_scale=1):
        super(DAU, self).__init__()
        ## Spatial Attention
        self.SA = spatial_attn_layer(n_feat)
        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

    def forward(self, x):
        ca_branch = self.CA(x)

        res = self.SA(ca_branch)

        res += x
        return res


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dilation=1):
        super(MakeDense, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Sequential(nn.ReflectionPad2d(self.padding),
                                  nn.Conv2d(in_channels=in_channels, out_channels=growth_rate, kernel_size=kernel_size,
                                            stride=1, padding=0, dilation=dilation, groups=1, bias=True))
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.act(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(in_channels=_in_channels, growth_rate=growth_rate, kernel_size=3, dilation=2 ** i))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)
        self.ca = ca_layer(in_channels)
        self.sa = spatial_attn_layer(in_channels, kernel_size=5)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        out = self.conv_1x1(out)

        out = self.sa(self.ca(out))

        out = out + x
        return out


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, 2, padding=0),

        nn.LeakyReLU(0.2, inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3, 1, padding=0),

        nn.LeakyReLU(0.2, inplace=True),
        DAU(out_channels)
    )


def first_double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(in_channels, out_channels, 7, 1, padding=0),

        nn.LeakyReLU(0.2, inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3, 1, padding=0),

        nn.LeakyReLU(0.2, inplace=True),
        DAU(out_channels)
    )


def up_double_conv(in_channels, out_channels):
    return nn.Sequential(
        DAU(in_channels),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, out_channels, 3, 1, padding=0),
        nn.LeakyReLU(0.2, inplace=True),
        nn.ReflectionPad2d(1),
        nn.Conv2d(out_channels, out_channels, 3, 1, padding=0),
        nn.LeakyReLU(0.2, inplace=True),
        DAU(out_channels)
    )


# =============== Mask Attention ===================
def mask_conv_sig(mask):
    return nn.Sequential(
        nn.ReflectionPad2d(3),
        nn.Conv2d(3, 1, 7, 2, padding=0),
        nn.Sigmoid()
    )


class Attention_Mask(nn.Module):
    def __init__(self, in_channel, mask):
        super(Attention_Mask, self).__init__()
        self.down_sample = mask_conv_sig(mask)
        self.convd3 = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(in_channel, 1, 3, 1, 0, 1))

    def forward(self, x):
        x1 = self.convd3(x)
        x = torch.cat((x1, torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x = self.down_sample(x)
        return x


def single_conv(in_channels, out_channels, k=3, s=1, p=0, d=1, active_function='ReLU'):
    if active_function == 'Tanh':
        af = nn.Tanh()
    elif active_function == 'ReLU':
        af = nn.LeakyReLU(0.2, inplace=True)
    elif active_function == 'Sigmoid':
        af = nn.Sigmoid()

    return nn.Sequential(
        nn.ReflectionPad2d(int((k - 1) / 2)),
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=s, padding=p, dilation=d),
        af
    )


class FeatureFusion(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(FeatureFusion, self).__init__()
        self.conv = up_double_conv(in_channel, out_channel)
        self.conv_global = single_conv(out_channel, out_channel, 3, 1, 0, 1, active_function='ReLU')
        self.conv_out1 = single_conv(out_channel, out_channel, 7, 1, 0, 1, active_function='ReLU')
        self.conv_out2 = single_conv(out_channel, out_channel, 3, 1, 0, 1, active_function='ReLU')

    def forward(self, over, under, all):
        x = torch.cat([under + over, all], dim=1)
        x = self.conv(x)
        g = self.conv_global(all)

        out = self.conv_out2(self.conv_out1(g+x))

        return out


class EIN(nn.Module):
    def __init__(self, num_c=32):
        super(EIN, self).__init__()

        self.dconv_down1_w = first_double_conv(3, num_c)
        self.dconv_down2_w = double_conv(num_c, num_c * 2)
        self.dconv_down3_w = double_conv(num_c * 2, num_c * 4)
        self.dconv_down4_w = double_conv(num_c * 4, num_c * 8)

        self.dconv_up4 = up_double_conv(num_c * 8, num_c * 4)
        self.dconv_up3 = up_double_conv(num_c * 6, num_c * 3)
        self.dconv_up2 = up_double_conv(num_c * 4, num_c * 2)
        self.dconv_up1 = up_double_conv(num_c * 2, num_c * 1)

        self.upsample3 = nn.Sequential(Interpolate(2, 'nearest', align_corners=None),
                                       nn.Conv2d(num_c * 4, num_c * 2, 1, 1, 0, 1, 1))
        self.upsample2 = nn.Sequential(Interpolate(2, 'nearest', align_corners=None),
                                       nn.Conv2d(num_c * 3, num_c * 2, 1, 1, 0, 1, 1))
        self.upsample1 = nn.Sequential(Interpolate(2, 'nearest', align_corners=None),
                                       nn.Conv2d(num_c * 2, num_c * 1, 1, 1, 0, 1, 1))

        self.featurefusion4 = FeatureFusion(num_c * 16, num_c * 8)
        self.featurefusion3 = FeatureFusion(num_c * 8, num_c * 4)
        self.featurefusion2 = FeatureFusion(num_c * 4, num_c * 2)
        self.featurefusion1 = FeatureFusion(num_c * 2, num_c)


        self.mask_down_4 = Attention_Mask(num_c * 4, 'down')
        self.mask_down_3 = Attention_Mask(num_c * 2, 'down')
        self.mask_down_2 = Attention_Mask(num_c, 'down')

        self.mask_up_4 = Attention_Mask(num_c * 4, 'up')
        self.mask_up_3 = Attention_Mask(num_c * 2, 'up')
        self.mask_up_2 = Attention_Mask(num_c, 'up')

        self.conv_last1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(num_c * 1, num_c * 1, 3, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_c * 1, 3, 7, 1, padding=0),
            nn.Sigmoid()
        )

        self.downsampler = Interpolate(scale_factor=1 / 2, mode='bilinear', align_corners=False)

        resblock = [RDB(num_c * 8, 4, num_c * 2) for i in range(3)]
        self.resblock = nn.Sequential(*resblock)

    def forward(self, ldr, hdr, mask_down, mask_up):
        ldr_low = ldr * mask_down
        ldr_high = ldr * mask_up

        mask_up = torch.mean(mask_up, dim=1, keepdim=True)
        mask_down = torch.mean(mask_down, dim=1, keepdim=True)

        low_img = ldr_low  # 3
        low_img_d1_1 = self.dconv_down1_w(low_img)
        low_img_d1_3 = low_img_d1_1 * mask_down  # num_c  256*256

        mask_down_2 = self.mask_down_2(low_img_d1_1) * (self.downsampler(mask_down) ** 2)
        low_img_d2_1 = self.dconv_down2_w(low_img_d1_3)
        low_img_d2_3 = low_img_d2_1 * mask_down_2  # 2num_c 128*128

        mask_down_3 = self.mask_down_3(low_img_d2_1) * (self.downsampler(mask_down_2) ** 2)
        low_img_d3_1 = self.dconv_down3_w(low_img_d2_3)
        low_img_d3_3 = low_img_d3_1 * mask_down_3  # 4num_c 64*64

        mask_down_4 = self.mask_down_4(low_img_d3_1) * (self.downsampler(mask_down_3) ** 2)
        low_img_d4_1 = self.dconv_down4_w(low_img_d3_3)
        low_img_d4_1 = self.resblock(low_img_d4_1)
        low_img_d4_3 = low_img_d4_1 * mask_down_4  # 8num_c 32*32

        # for the high dynamic range
        high_img = ldr_high  # 3
        high_img_d1_1 = self.dconv_down1_w(high_img)
        high_img_d1_3 = high_img_d1_1 * mask_up  # num_c  256*256

        mask_up_2 = self.mask_up_2(high_img_d1_1) * (self.downsampler(mask_up) ** 2)
        high_img_d2_1 = self.dconv_down2_w(high_img_d1_3)
        high_img_d2_3 = high_img_d2_1 * mask_up_2  # 2num_c 128*128

        mask_up_3 = self.mask_up_3(high_img_d2_1) * (self.downsampler(mask_up_2) ** 2)
        high_img_d3_1 = self.dconv_down3_w(high_img_d2_3)
        high_img_d3_3 = high_img_d3_1 * mask_up_3  # 4num_c 64*64

        mask_up_4 = self.mask_up_4(high_img_d3_1) * (self.downsampler(mask_up_3) ** 2)
        high_img_d4_1 = self.dconv_down4_w(high_img_d3_3)
        # high_img_d4_1 = self.resblock(high_img_d4_1) + high_img_d4_1
        high_img_d4_1 = self.resblock(high_img_d4_1)
        high_img_d4_3 = high_img_d4_1 * mask_up_4  # 8num_c 32*32

        # for the whole dynamic range
        whole_img_d1 = self.dconv_down1_w(ldr)  # num_c  256*256
        whole_img_d2 = self.dconv_down2_w(whole_img_d1)  # 2num_c 128*128
        whole_img_d3 = self.dconv_down3_w(whole_img_d2)  # 4num_c 64*64
        whole_img_d4 = self.dconv_down4_w(whole_img_d3)  # 8num_c 32*32
        # whole_img_d4 = self.resblock(whole_img_d4) + whole_img_d4
        whole_img_d4 = self.resblock(whole_img_d4)

        # for upsample
        x = self.featurefusion4(high_img_d4_3, low_img_d4_3, whole_img_d4)  # 8*3num_c 32*32 - 24-8
        x = self.dconv_up4(x)  # 8num_c 32*32

        x = self.upsample3(x)  # 8num_c 64*64
        x = torch.cat([x, self.featurefusion3(high_img_d3_3, low_img_d3_3, whole_img_d3)], dim=1)  # 8+4num_c  64*64
        x = self.dconv_up3(x)  # 6num_c

        x = self.upsample2(x)  # 6num_c  128*128
        x = torch.cat([x, self.featurefusion2(high_img_d2_3, low_img_d2_3, whole_img_d2)], dim=1)  # 6+2num_c  128*128
        x = self.dconv_up2(x)  # 4num_c

        x = self.upsample1(x)  # 4num_c  256*256
        x = torch.cat([x, self.featurefusion1(high_img_d1_3, low_img_d1_3, whole_img_d1)], dim=1)  # 4+1num_c  256*256
        x = self.dconv_up1(x)  # 2num_c  256*256

        out = self.conv_last1(x)  # num_c

        return out
