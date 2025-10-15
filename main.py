import argparse
import torch
torch.cuda.empty_cache()
import trainer as trainer
import os
import ast
from model import EIN
from utils import mk_dir
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


# ----------------------------------------
# Parameter initialization
# ----------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # training settings
    parser.add_argument("--train", type=ast.literal_eval, default=False)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument('--pre_model', default=None,
                        help='used for resume training.')
    parser.add_argument('--test_model', default=r'./epoch_146.pth',
                        help='used for resume training.')
    parser.add_argument('--start_epoch', default=0, help='used for resume training.')
    parser.add_argument('--max_epoch', default=200, help='the max epoch for training.')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate for training.')
    parser.add_argument('--train_batch_size', type=int, default=8, help='Train batch size.')
    parser.add_argument('--valid_batch_size', type=int, default=1, help='Valid batch size.')
    parser.add_argument("--crop_size", type=int, default=(224, 224))
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--loss_freq', type=int, default=10, help='Report (average) loss every x iterations.')
    parser.add_argument('--checkpoints_freq', type=int, default=1, help='save the checkpoints every x epoch.')
    parser.add_argument('--valid_freq', type=int, default=1, help='validate the results every x epoch.')
    parser.add_argument('--check_img', type=int, default=10000,
                        help='show the images during training stage every x iterations.')
    parser.add_argument('--hdr_ext', default=['.hdr', '.exr'])
    parser.add_argument('--sdr_ext', default=['.png', '.jpg', '.jpeg', '.tif'])
    parser.add_argument('--init_type', type=str, default='kaiming')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')

    # path of training data
    parser.add_argument('--root_path', default=r'./data', help='Path to data.')
    parser.add_argument('--train_file', default=r'train.xls', help='Excel path of training data.')
    parser.add_argument('--valid_file', default=r'valid.xls', help='Excel path of valid data.')
    parser.add_argument('--test_SDR_path', default=r'./test_SDR', help='Path of testing SDR images.')
    parser.add_argument('--out_HDR_path', default=r'./test_results',
                        type=str, metavar='PATH',
                        help='Root path for results.')

    # path of results
    parser.add_argument('--results_path', default='./results/', type=str, metavar='PATH', help='Root path for results.')
    parser.add_argument('--log_path', default='./logs/', type=str, metavar='PATH', help='Path for logs file.')
    parser.add_argument('--ckpt_path', default='./ckpt/', type=str, metavar='PATH',
                        help='Path for checkpoints.')

    return parser.parse_args()


def main(opt):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    print('Train: ', opt.train)
    T = trainer.Trainer(opt)

    if opt.train:
        T.train()
    else:
        print('predicted')
        write_path = opt.out_HDR_path
        mk_dir(write_path)
        PATH = opt.test_model  # i_3: 259 / 330

        print(write_path)
        print(PATH)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        net = EIN()
        tmp = torch.load(PATH, map_location=device)
        # replace the module. to ''
        tmp2 = {k.replace('module.', ''): v for k, v in tmp.items()}
        net.load_state_dict(tmp2)
        net.eval()

        T.test(net, write_path)


if __name__ == "__main__":
    opt = parse_args()

    mk_dir(opt.results_path)
    mk_dir(os.path.join(opt.results_path, opt.log_path))
    mk_dir(os.path.join(opt.results_path, opt.ckpt_path))

    main(opt)


# python main.py --train False --test_SDR_path ./test_SDR --out_HDR_path ./test_results
# python main.py --train True --root_path ./data

