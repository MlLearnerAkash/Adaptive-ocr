import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import lmdb
import torchvision.transforms as transforms
import six
import sys
from PIL import Image
import numpy as np
import os
import sys
import pdb
import pickle

class PickleDataset(Dataset):
    def __init__(self, opt, mode = "train"):
        super(PickleDataset, self).__init__()
        self.mode = mode
        
        if self.mode == "train":
            pickle_file = os.path.join(opt.path, opt.imgdir, '%s.data.pkl'%opt.language)
            with open(pickle_file, 'rb') as f:
                self.data = pickle.load(f)
            self.nSamples = len(self.data['train'])
        else:
            pickle_file = os.path.join(opt.path, opt.val_dir, '%s.data.pkl'%opt.language)
            with open(pickle_file, 'rb') as f:
                self.data = pickle.load(f)
            self.nSamples =len(self.data["val"])
        transform_list =  [transforms.Grayscale(1),
                            transforms.ToTensor(), 

                            transforms.Normalize((0.5,), (0.5,)),
                            transforms.Resize((32,100))
                            ]
        self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        if self.mode == "train":
            img, label = self.data['train'][index]
        else:
            img, label = self.data['val'][index]
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        item = {'img': img, 'idx':index}
        item['label'] = label
        return item 
