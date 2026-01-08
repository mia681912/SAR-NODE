"""
read dataset from .h5 file
maziq
2024/08/27
"""


import torch.utils.data as data
import torch
import h5py
import numpy as np


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)
        # print(dataset.data)

        self.clean = dataset["clean"]
        self.noise = dataset["noise"]


    def __getitem__(self, index):
        self.img_clean = self.clean.get('{}'.format(index))[:,:,:,:]
        self.img_clean1 = np.squeeze(self.img_clean, 0)
        self.img_noise = self.noise.get('{}'.format(index))[:,:,:,:]
        self.img_noise1 = np.squeeze(self.img_noise, 0)
        return self.img_clean1, \
               self.img_noise1


    def __len__(self):
        return len(self.clean)
