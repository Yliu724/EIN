import os
import cv2
import pandas as pd
from torch.utils.data import Dataset

class SDRHDRDataset(Dataset):
    def __init__(
            self,
            root_path='./data',
            xls_file='train.xls',
            data_type='train',
            crop_size=None,
            hdr_extensions=None,
            ldr_extensions=None,
            preprocess=None,
    ):
        super(SDRHDRDataset, self).__init__()

        if hdr_extensions is None:
            hdr_extensions = ['.hdr', '.exr']

        if ldr_extensions is None:
            ldr_extensions = ['.png', '.jpg']

        if data_type == 'train':
            xl_file_path = os.path.join(root_path, xls_file)
            writer = pd.ExcelFile(xl_file_path)
            data_frame = writer.parse(data_type)
            print(xl_file_path, data_type)
            self.img_name = data_frame['HDR'].values.tolist()  # HDR name
            self.sdr_name = data_frame['SDR'].values.tolist()  # HDR name
            if len(self.img_name) == 0:
                msg = 'Could not find any files with ext:\n[{0}]'
                raise RuntimeError(
                    msg.format(', '.join(hdr_extensions))
                )

        if data_type == 'valid':
            xl_file_path = os.path.join(root_path, xls_file)
            writer = pd.ExcelFile(xl_file_path)
            data_frame = writer.parse('valid')
            self.img_name = data_frame['HDR'].values.tolist()  # HDR name
            self.sdr_name = data_frame['SDR'].values.tolist()  # HDR name
            if len(self.img_name) == 0:
                msg = 'Could not find any files with extensions:\n[{0}]'
                raise RuntimeError(
                    msg.format(', '.join(hdr_extensions))
                )

        if data_type == 'test':
            self.img_name = []
            for root, dirs, files in sorted(os.walk(root_path)):
                for file in files:
                    if any(
                            file.lower().endswith(extension)
                            for extension in ldr_extensions
                    ):
                        self.img_name.append(file)

            if len(self.img_name) == 0:
                msg = 'Could not find any files with extensions:\n[{0}]'
                raise RuntimeError(
                    msg.format(', '.join(ldr_extensions))
                )

        self.data_type = data_type
        self.preprocess = preprocess
        self.root_path = root_path
        self.crop_size = crop_size

    def __getitem__(self, index):
        if not self.data_type == 'test':
            sdr_path = os.path.join(self.root_path, self.data_type + '_SDR', self.sdr_name[index])
            hdr_path = os.path.join(self.root_path, self.data_type + '_HDR', self.img_name[index])

            if not os.path.exists(sdr_path):
                print('sdr not exists!!!')
                print(sdr_path)
            if not os.path.exists(hdr_path):
                print('hdr not exists!!!')
                print(hdr_path)

            sdr = cv2.imread(sdr_path)
            hdr = cv2.imread(hdr_path, cv2.IMREAD_UNCHANGED)
            if self.preprocess is not None:
                dpoint = self.preprocess(hdr, sdr, self.crop_size, os.path.split(sdr_path)[-1])


        if self.data_type == 'test':
            sdr_path = os.path.join(self.root_path, self.img_name[index])
            sdr = cv2.imread(sdr_path)
            if self.preprocess is not None:
                dpoint = self.preprocess(sdr, self.crop_size, os.path.split(sdr_path)[-1])
        return dpoint

    def __len__(self):
        return len(self.img_name)
