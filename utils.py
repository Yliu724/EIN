import os
import numpy as np
from numpy.random import uniform
import torch
import cv2


def mk_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def __make_power_32(img, base=32, long=512):
    ow_, oh_ = img.shape[0], img.shape[1]

    if ow_ >= oh_:
        oh = int(oh_ * long / ow_)
        ow = long
    else:
        ow = int(ow_ * long / oh_)
        oh = long

    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)

    if (h == oh_) and (w == ow_):
        return img
    else:
        return cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)


def print_network(model, name):
    """Print out the network information."""
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    print("=== The number of parameters of model [{}] is [{}] or [{:>.4f}M]".format(name, num_params, num_params / 1e6))


def random_flip(x, y):
    rot = np.random.randint(-1, 2)
    return cv2.flip(x, rot), cv2.flip(y, rot)


def inverse_tone_mapping(x):

    return x ** (2.2)


def Gaussian_up(img, mu=0.95, sigma=0.01):
    return np.exp(-(img - mu) ** 2 / sigma)


def Gaussian_down(img, mu=0.05, sigma=0.01):
    return np.exp(-(img - mu) ** 2 / sigma)


def img_segmentation2(ldr):
    hdr_in = inverse_tone_mapping(ldr)
    ldr_gray = cv2.cvtColor(ldr, cv2.COLOR_BGR2GRAY)
    ldr_gray = map_range(ldr_gray.astype('float32'))

    # ==== up and down with gaussian
    mask_up = Gaussian_up(ldr_gray, mu=1.0)
    mask_down = Gaussian_down(ldr_gray, mu=0)

    mask_down = np.array([mask_down, mask_down, mask_down]).transpose((1, 2, 0))
    mask_up = np.array([mask_up, mask_up, mask_up]).transpose((1, 2, 0))
    kernel = np.ones((3, 3), np.uint8)
    mask_down = cv2.dilate(mask_down, kernel, iterations=1)
    mask_up = cv2.dilate(mask_up, kernel, iterations=1)

    return hdr_in.astype('float32'), mask_down.astype('float32'), mask_up.astype('float32')


def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)


def to_np_filp(p):
    r = p.squeeze(0)
    r = r.transpose(0, 1)
    r = r.transpose(1, 2)
    r = r.detach().numpy()[:, :, (2, 1, 0)]
    return r


def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]  # bgr to rgb
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))


class BaseTMO(object):
    def __call__(self, img):
        return self.op.process(img)


class Reinhard(BaseTMO):
    def __init__(
            self,
            intensity=-1.0,
            light_adapt=0.8,
            color_adapt=0.0,
            gamma=2.0,
            randomize=False,
    ):
        if randomize:
            gamma = uniform(1.8, 2.2)
            intensity = uniform(-1.0, 1.0)
            light_adapt = uniform(0.8, 1.0)
            color_adapt = uniform(0.0, 0.2)
        self.op = cv2.createTonemapReinhard(
            gamma=gamma,
            intensity=intensity,
            light_adapt=light_adapt,
            color_adapt=color_adapt,
        )


TRAIN_TMO_DICT = {
    'reinhard': Reinhard,
}

def random_tone_map(x):
    x = map_range(x)
    tmos = list(TRAIN_TMO_DICT.keys())
    tmo = TRAIN_TMO_DICT[tmos[0]](randomize=False)
    return map_range(tmo(x))
