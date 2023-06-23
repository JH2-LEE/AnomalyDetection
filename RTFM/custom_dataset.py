import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
torch.set_default_tensor_type('torch.FloatTensor')

import option
from config import *


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False, dataset=None):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        # self.dataset = dataset #model2
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        elif self.dataset == 'ccd':
            if test_mode:
                self.rgb_list_file = 'list/ccd-test-i3d-10crop.list'
            else:
                self.rgb_list_file = 'list/ccd-test-i3d-10crop.list'
        elif self.dataset == 'carla':
            if test_mode:
                self.rgb_list_file = 'list/carla-test-i3d-10crop.list'
            else:
                self.rgb_list_file = 'list/carla-test-i3d-10crop.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'

        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)
            elif self.dataset == 'ccd':
                if self.is_normal:
                    self.list = list(open('list/ccd-train-normal-i3d-10crop.list'))
                    print('normal list for car crash')
                    print(self.list)
                else:
                    self.list = list(open('list/ccd-train-abnormal-i3d-10crop.list'))
                    print('abnormal list for car crash')
                    print(self.list)
            elif self.dataset == 'carla':
                if self.is_normal:
                    self.list = list(open('list/carla-train-normal-i3d-10crop.list'))
                    print('normal list for carla')
                    print(self.list)
                else:
                    self.list = list(open('list/carla-train-abnormal-i3d-10crop.list'))
                    print('abnormal list for carla')
                    print(self.list)
            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)

    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)

        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            # print("0",features.shape)
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            # print("1",features.shape)
            divided_features = []
            for feature in features:
                # print("2",feature.shape)
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                # print("3",feature.shape)
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)
            # print('5', divided_features.shape)
            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame

if __name__ == '__main__':
    args = option.parser.parse_args()
    config = Config(args)
    print(args)
    train_nloader = DataLoader(Dataset(args, test_mode=False, is_normal=True),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    train_aloader = DataLoader(Dataset(args, test_mode=False, is_normal=False),
                               batch_size=args.batch_size, shuffle=True,
                               num_workers=0, pin_memory=False, drop_last=True)
    test_loader = DataLoader(Dataset(args, test_mode=True),
                              batch_size=1, shuffle=False,
                              num_workers=0, pin_memory=False)
    for i, batch in enumerate(test_loader):
        # features, labels = batch
        features = batch
        print(features.size())
        # print(labels.size())
        # print(labels)
        exit()