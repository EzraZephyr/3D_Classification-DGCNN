import os
import h5py
import numpy as np
from torch.utils.data import Dataset


def load_h5data(filename):
    with h5py.File(filename, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]

    f.close()
    return data, label

class ModelNet400Dataset(Dataset):

    def __init__(self, data_dir, split='train', num_points=1024):
        self.data_dir = data_dir
        self.split = split
        self.num_points = num_points

        if self.split == 'train':
            file_path = os.path.join(self.data_dir, 'train_files.txt')
        else:
            file_path = os.path.join(self.data_dir, 'test_files.txt')

        with open(file_path, 'r') as f:
            self.f5_files = [line.strip() for line in f]

        self.data, self.labels = self.load_data()

    def load_data(self):
        all_data = []
        all_labels = []
        for h5_file in self.f5_files:
            data, label = load_h5data(h5_file)
            all_data.append(data)
            all_labels.append(label)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        return all_data, all_labels.squeeze()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        point = self.data[idx]
        choice = np.random.choice(point.shape[0], self.num_points, replace=False)
        points = point[choice, :]
        # 这里如果设备允许的话最好把2048个点全选上 或者进行最远点采样

        label = self.labels[idx]

        return points, label
      