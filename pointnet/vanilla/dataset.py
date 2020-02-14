import torch
import torch.utils.data as data
import os
import os.path
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement

def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)

    #print('classes: ', classes)

    return classes


class ModelNetDataset(data.Dataset):
    def __init__(self, root, n_pts=2500, split='train'):
        self.n_pts = n_pts
        self.root = root
        self.split = split
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip().split('/'))

        #self.fns = np.unique(self.fns)

        self.cat = {}
        self.classes = gen_modelnet_id(root)
        for i in range (len(self.classes)):
            self.cat[self.classes[i]] = i

        print('cat: \n', self.cat)

    def __getitem__(self,index):
        fn = self.fns[index]
        #print('fn: \n', fn, '\n')
        cls = self.cat[fn[0]]
        #print('cls: \n', cls, '\n')
        return cls

    def __len__(self):
        return len(self.fns)

if __name__ == '__main__':
    datapath = sys.argv[1]

    classes = gen_modelnet_id(datapath)
    print('classes: \n', classes, '\nnumber of classes: \n', len(classes))

    data = ModelNetDataset(root=datapath)
    print('number of data: ', len(data))
    print('the first item is in class {}'.format(data[0]))