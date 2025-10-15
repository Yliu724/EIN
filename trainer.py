import os
import time
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import torch
torch.cuda.empty_cache()
import numpy as np
from torch.utils.data import DataLoader
from losses import EINLoss, PSNR
from ImageDataset import SDRHDRDataset
from utils import (
    mk_dir,
    random_flip,
    map_range,
    img_segmentation2,
    cv2torch,
    __make_power_32,
    print_network,
    random_tone_map,
    to_np_filp,
)
from model import EIN, init_weights


def gamma_trans(img_in):
    param = 1 / 6.5
    out = torch.pow(img_in, param)
    print(img_in.max(), img_in.min())
    return out


def lambda_trans(input):
    mu = 5000
    return torch.log10(1. + mu * input) / np.log10(1. + mu)



# ----------------------------------------
# Transform function
# ----------------------------------------
def train_transform(hdr, ldr, crop_size, name):
    hdr_gt = cv2.resize(hdr, crop_size, interpolation=cv2.INTER_AREA)
    ldr_in = cv2.resize(ldr, crop_size, interpolation=cv2.INTER_AREA)
    hdr_gt, ldr_in = random_flip(hdr_gt, ldr_in)
    hdr_gt = map_range(hdr_gt)
    hdr_in, mask_down, mask_up = img_segmentation2(ldr_in)
    ldr_in = ldr_in / 255.0
    ldr_in = ldr_in.astype('float32')
    return cv2torch(ldr_in), cv2torch(hdr_in), cv2torch(mask_down), cv2torch(mask_up), cv2torch(hdr_gt)


def valid_transform(hdr, ldr, crop_size, name):
    hdr_gt = __make_power_32(hdr, long=640)
    ldr_in = cv2.resize(ldr, (hdr_gt.shape[1], hdr_gt.shape[0]), interpolation=cv2.INTER_AREA)
    hdr_gt = map_range(hdr_gt)
    hdr_in, mask_down, mask_up = img_segmentation2(ldr_in)
    ldr_in = ldr_in / 255.0
    ldr_in = ldr_in.astype('float32')
    return cv2torch(ldr_in), cv2torch(hdr_in), cv2torch(mask_down), \
           cv2torch(mask_up), cv2torch(hdr_gt), name


def test_transform(ldr, crop_size, name):
    ldr_in = __make_power_32(ldr, long=2048)
    h, w = ldr.shape[0], ldr.shape[1]
    hdr_in, mask_down, mask_up = img_segmentation2(ldr_in)
    ldr_in = ldr_in / 255.0
    ldr_in = ldr_in.astype('float32')
    return cv2torch(ldr_in), cv2torch(hdr_in), cv2torch(mask_down), \
           cv2torch(mask_up), name, w, h


class Trainer(object):
    def __init__(self, opt):
        torch.manual_seed(opt.seed)

        self.opt = opt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)

        self.train_transform = train_transform
        self.valid_transform = valid_transform
        self.test_transform = test_transform

        self.train_batch_size = opt.train_batch_size
        self.valid_batch_size = opt.valid_batch_size

        if self.opt.train:

            self.train_data = SDRHDRDataset(root_path=opt.root_path,
                                            xls_file=opt.train_file,
                                            data_type='train',
                                            crop_size=opt.crop_size,
                                            hdr_extensions=opt.hdr_ext,
                                            ldr_extensions=opt.sdr_ext,
                                            preprocess=self.train_transform)
            self.train_loader = DataLoader(self.train_data,
                                           batch_size=self.train_batch_size,
                                           num_workers=opt.num_workers,
                                           shuffle=True,
                                           drop_last=True,
                                           pin_memory=True)

            self.valid_data = SDRHDRDataset(root_path=opt.root_path,
                                            xls_file=opt.valid_file,
                                            data_type='valid',
                                            hdr_extensions=opt.hdr_ext,
                                            ldr_extensions=opt.sdr_ext,
                                            preprocess=self.valid_transform)
            self.valid_loader = DataLoader(self.valid_data,
                                           batch_size=self.valid_batch_size,
                                           shuffle=False,
                                           pin_memory=True,
                                           num_workers=opt.num_workers)
        else:
            self.test_data = SDRHDRDataset(root_path=opt.test_SDR_path,
                                           xls_file=opt.test_SDR_path,
                                           data_type='test',
                                           hdr_extensions=opt.hdr_ext,
                                           ldr_extensions=opt.sdr_ext,
                                           preprocess=self.test_transform)
            self.test_loader = DataLoader(self.test_data,
                                          batch_size=1,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=opt.num_workers)

        # initialize the model
        self.EIN = EIN().to(self.device)
        init_weights(self.EIN, init_type=opt.init_type, gain=0.02)
        self.EIN_name = type(self.EIN).__name__
        print_network(self.EIN, self.EIN_name)

        self.train_criterion = EINLoss().to(self.device)
        self.val_criterion = PSNR()

        self.optimizer = torch.optim.Adam(self.EIN.parameters(), lr=opt.lr, betas=(0.95, 0.999))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=25, eta_min=1e-7)

        self.avg_total_loss = 0
        self.avg_percep_loss = 0
        self.avg_l2_loss = 0
        self.avg_cosine_loss = 0
        self.valid_freq = opt.valid_freq
        self.check_img = opt.check_img
        self.loss_freq = opt.loss_freq
        self.start_epoch = opt.start_epoch
        self.max_epoch = opt.max_epoch
        self.crop_size = opt.crop_size
        self.checkpoints_freq = opt.checkpoints_freq

        self.val_interval = 1
        self.valid_loss = []

        if opt.pre_model:
            print('load pre-model')

            tmp = torch.load(opt.pre_model, map_location='cuda:0')
            checkpoint = {k.replace('module.', ''): v for k, v in tmp.items()}
            self.EIN.load_state_dict(checkpoint)  # 加载模型可学习参数

        self.results_path = opt.results_path
        self.ckpt_path = os.path.join(self.results_path, opt.ckpt_path)
        self.log_path = os.path.join(self.results_path, opt.log_path)

        if not os.path.exists(os.path.join(opt.results_path, opt.ckpt_path)):
            os.mkdir(opt.save_path)
        else:
            print(
                'WARNING: save_path already exists. '
                'Checkpoints may be overwritten'
            )

    def train(self):
        iter_count = 0
        self.pbar = tqdm(total=self.max_epoch * len(self.train_loader), desc='Train epoches', position=0)
        self.pbar.write("=========== Start training ===========")

        for epoch in range(self.start_epoch, self.max_epoch):
            self.EIN.train()

            # initialize logging system
            num_steps_per_epoch = len(self.train_loader)
            self.pbar.write("== epoch [{}], num_steps_per_epoch: [{}], ".format(epoch + 1, num_steps_per_epoch), end='')
            local_counter = epoch * num_steps_per_epoch + 1
            start_time = time.time()

            # start training
            for param_group in self.optimizer.param_groups:
                self.pbar.write("Adam learning rate: [{:>1.7f}] ==".format(param_group['lr']))

            for step, (ldr_in, hdr_in, mask_down, mask_up, hdr_target) in enumerate(self.train_loader, 0):
                iter_count += 1
                self.optimizer.zero_grad()

                ldr_in = ldr_in.to(self.device)
                hdr_in = hdr_in.to(self.device)
                mask_down = mask_down.to(self.device)
                mask_up = mask_up.to(self.device)
                hdr_target = hdr_target.to(self.device)

                hdr_prediction = self.EIN(ldr_in, hdr_in, mask_down, mask_up)


                self.train_loss, self.percep_term, self.l2_term, self.cosine_term = self.train_criterion(hdr_prediction, hdr_target)

                self.train_loss.backward()
                self.optimizer.step()

                # print statistics
                self.avg_total_loss += self.train_loss.item()
                self.avg_percep_loss += self.percep_term.item()
                self.avg_l2_loss += self.l2_term.item()
                self.avg_cosine_loss += self.cosine_term.item()
                if ((step + 1) % self.loss_freq) == 0:

                    rep = (
                        f'Epoch: {epoch + 1:>5d}, '
                        f'Iter: {step + 1:>6d}, '
                        f'Total Loss: {self.train_loss.item():>3.3f}, '
                        f' | Percep: {self.percep_term.item():>3.3f}, '
                        f'L2: {self.l2_term.item():>3.3f}, '
                        f'Cos: {self.cosine_term.item():>3.3f}, '
                    )
                    tqdm.write(rep)
                    self.avg_total_loss = 0
                    self.avg_percep_loss = 0
                    self.avg_l2_loss = 0
                    self.avg_cosine_loss = 0

                if ((step + 1) % 100) == 0:
                    torch.save(self.EIN.state_dict(), os.path.join(self.ckpt_path, f'epoch_{epoch + 1}.pth'), )

                local_counter += 1
                self.pbar.update(1)

                del self.train_loss, self.percep_term, self.l2_term, self.cosine_term

            current_time = time.time()
            duration = current_time - start_time
            self.pbar.write("== Epoch {} takes {:>4.2f} mins ==".format(epoch + 1, duration / 60))

            self.scheduler.step()  # 更新学习率

            if (epoch + 1) % self.valid_freq == 0:
                test_results, sec_per_sample = self.eval(epoch)

                f = open(os.path.join(self.results_path, 'valid_loss.txt'), 'a')
                f.write('Epoch ' + str(epoch + 1) + ': ' + str(test_results) + '\n')
                f.close()

                self.pbar.write(
                    "== Epoch {} Validation: Average PU-PSNR = {:>3.4f}, ({:>1.2f} sec/sample) ==".format(
                        epoch + 1, test_results, sec_per_sample))

            if (epoch % self.checkpoints_freq) == 0:
                torch.save(
                    self.EIN.state_dict(),
                    os.path.join(self.ckpt_path, f'epoch_{epoch + 1}.pth'),
                )

        self.pbar.write("=========== Complete training ===========")

    def eval(self, epoch):
        val_pbar = tqdm(total=len(self.valid_loader), desc='Validation')

        val_PSNR_TM = []
        start_time = time.time()
        with torch.no_grad():
            for step, (ldr_in, hdr_in, mask_down, mask_up, hdr_target, img_name) in enumerate(self.valid_loader):
                self.EIN.eval()

                ldr_in = ldr_in.cuda(non_blocking=True)
                hdr_in = hdr_in.cuda(non_blocking=True)
                mask_down = mask_down.cuda(non_blocking=True)
                mask_up = mask_up.cuda(non_blocking=True)
                hdr_target = hdr_target.cuda(non_blocking=True)
                img_name = img_name[0]

                hdr_prediction = self.EIN(ldr_in, hdr_in, mask_down, mask_up)

                TM_hdr_prediction = map_range(random_tone_map(to_np_filp(hdr_prediction.cpu())), 0, 255).astype('uint8')
                TM_hdr_target = map_range(random_tone_map(to_np_filp(hdr_target.cpu())), 0, 255).astype('uint8')

                tm_psnr = self.val_criterion(TM_hdr_prediction, TM_hdr_target, 255.0)
                val_PSNR_TM.append(tm_psnr)

                val_pbar.update(1)

            torch.cuda.empty_cache()
            current_time = time.time()
            duration = current_time - start_time
            sec_per_sample = duration / (step + 1)

            avg_psnr_tm = sum(val_PSNR_TM) / len(val_PSNR_TM)

        return avg_psnr_tm, sec_per_sample



    def test(self, EIN_net, write_path):

        pred_HDR_path = write_path
        mk_dir(pred_HDR_path)

        EIN_net.to(self.device)

        avg_time = 0
        with torch.no_grad():
            for step, (ldr_in, hdr_in, mask_down, mask_up, img_name, w, h) in enumerate(self.test_loader):
                print(step, 'predict - ', img_name[0], '...')
                ldr_in = ldr_in.to(self.device)
                hdr_in = hdr_in.to(self.device)
                mask_down = mask_down.to(self.device)
                mask_up = mask_up.to(self.device)
                img_name = img_name[0]

                start_time = time.time()

                hdr_prediction = EIN_net(ldr_in, hdr_in, mask_down, mask_up)

                current_time = time.time()
                duration = current_time - start_time
                avg_time = avg_time + duration
                print('test time for one image: ', duration)

                w = w.numpy()[0]
                h = h.numpy()[0]
                cv2.imwrite(os.path.join(pred_HDR_path, img_name[:-4] + '_pre.hdr'), cv2.resize(to_np_filp(hdr_prediction.cpu()), (w, h)))

            torch.cuda.empty_cache()

        print(avg_time / step)