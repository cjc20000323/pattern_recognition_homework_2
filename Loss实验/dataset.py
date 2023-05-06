import sys
import os
import os.path as osp
from PIL import Image

import torch
import numpy as np
from sklearn.model_selection import train_test_split
from loguru import logger
from torch.utils.data import Dataset as torchDataset


class CASIA:

    def __init__(self, root) -> None:
        database = []
        label_ids = []
        dirs = os.listdir(root)
        next_label_id = 0
        for dir in dirs:
            img_files = os.listdir(osp.join(root, dir))
            for img_f in img_files:
                img_path = osp.join(root, dir, img_f)
                database.append((img_path, next_label_id))
                label_ids.append(next_label_id)
            next_label_id += 1
        idx = range(len(database))
        # train_idx, test_idx = train_test_split(idx, train_size=0.8, stratify=label_ids, random_state=0)
        
        train_idx = []
        test_idx = []
        with open("./checkpoint/base_with_crop_center的测试样本索引.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            l = line.strip()
            test_idx.append(int(l))

        with open("./trainidx.txt", "r") as f:
            lines = f.readlines()
        for line in lines:
            l = line.strip()
            train_idx.append(int(l))
        print(len(train_idx))
        trainset = set(train_idx)
        testset = set(test_idx)
        print("===", trainset & testset)
        
        # for i in idx:
        #     if i not in test_idx:
        #         train_idx.append(i)
        # print(len(idx), len(train_idx))
        # with open("trainidx.txt", "w") as f:
        #     for i in train_idx:
        #         f.write("{}\n".format(i))
        # assert 1 == 2

        self.train = [database[i] for i in train_idx]
        self.test = [database[i] for i in test_idx]


class Dataset(torchDataset):

    def __init__(self, database, transform=None):
        self.database = database
        self.transform = transform

    def __len__(self):
        return len(self.database)
    
    def __getitem__(self, index):
        img, pid = self.database[index]
        if isinstance(img, str):
            if not osp.exists(img):
                logger.error("Image {} does not exist!".format(img))
                sys.exit()
            img = Image.open(img).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, pid
    

def collate_fn(batch):
    imgs, pids = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


__factory = {
    "casia": CASIA,
}

def load_database(name, *args, **kwargs):
    if name not in __factory.keys():
        raise ValueError("Unknown database: {}".format(name))
    return __factory[name](*args, **kwargs)


def make_dataset(name, transform, *args, **kwargs):
    database = load_database(name, *args, **kwargs)
    train_dataset = Dataset(database.train, transform)
    test_dataset = Dataset(database.test, transform)
    return database, train_dataset, test_dataset