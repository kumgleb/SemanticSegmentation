import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self,
                 path: str,
                 transforms=None):
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.data_list = os.listdir(path)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        path_to_img = os.path.join(self.path, self.data_list[idx])
        img = np.asanyarray(Image.open(path_to_img), dtype='uint8')

        x = img[:, :256, :]
        y = img[:, 256:, :]

        if self.transforms:
            x, y = self.transforms(x, y)

        x = torch.from_numpy(x).type(torch.float32)
        y = torch.from_numpy(y).type(torch.long)

        return x, y
