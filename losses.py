import torchvision.models as models
import torch.nn as nn
import numpy as np
import torch


class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2, max_value):
        TM_mse = np.mean((img1 / 1. - img2 / 1.) ** 2)
        TM_PSNR = 20 * np.log10(max_value / np.sqrt(TM_mse))
        return TM_PSNR


class VGG19_relu(torch.nn.Module):
    def __init__(self):
        super(VGG19_relu, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        cnn = models.vgg19(pretrained=True)
        cnn = cnn.to(self.device)
        features = cnn.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out


class PerceptualLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.add_module('vgg', VGG19_relu())
        self.criterion = torch.nn.MSELoss()
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def __call__(self, x, y):
        if x.shape[1] != 3:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x = (x - self.mean.to(x)) / self.std.to(x)
        y = (y - self.mean.to(y)) / self.std.to(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        # # features after relu
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss / 5.


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


class MultiscaleRecLoss(nn.Module):
    def __init__(self, scale=4, rec_loss_type='l2', multiscal=True):
        super(MultiscaleRecLoss, self).__init__()
        if rec_loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif rec_loss_type == 'smoothl1':
            self.criterion = nn.SmoothL1Loss()
        elif rec_loss_type == 'l2':
            self.criterion = nn.MSELoss()
        self.downsample = Interpolate(scale_factor=1 / 2, mode='bilinear', align_corners=False)
        if multiscal:
            self.weights = [1, 1.0 / 2, 1.0 / 4, 1.0 / 6]
        else:
            self.weights = [1]
        self.weights = self.weights[:scale]

    def forward(self, input, target):
        loss = 0
        for i in range(len(self.weights)):
            loss += self.weights[i] * self.criterion(input, target)
            if i != len(self.weights) - 1:
                input = self.downsample(input)
                target = self.downsample(target)
        return loss


class EINLoss(nn.Module):
    def __init__(self):
        super(EINLoss, self).__init__()
        self.similarity = torch.nn.CosineSimilarity(dim=1, eps=1e-20)
        self.l2_loss = MultiscaleRecLoss(3, 'l1', True)
        self.percepLoss = PerceptualLoss()

        self.param_l2 = 10
        self.param_precep = 1
        self.param_cosine = 180 / np.pi

        self.param_gl = 1.0

        self.param_u = 1.0
        self.param_d = 1.0
        self.param_m = 1.0

    def lambda_trans(self, input):
        mu = 5000
        return torch.log(1. + mu * input) / np.log(1. + mu)

    def forward(self, x, y, trans_mode='mu'):
        x_trans_mu = self.lambda_trans(x)
        y_trans_mu = self.lambda_trans(y)

        l2_term_m = self.param_l2 * self.l2_loss(x_trans_mu, y_trans_mu).mean()
        percep_term_m = self.param_precep * self.percepLoss(x_trans_mu, y_trans_mu)
        cosine_term_m = self.param_cosine * (1 - self.similarity(x_trans_mu, y_trans_mu)).mean()

        loss_m = percep_term_m + l2_term_m + cosine_term_m
        out = loss_m

        return out, percep_term_m, l2_term_m, cosine_term_m
