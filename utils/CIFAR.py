import os
from PIL import Image
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


import os
from PIL import Image
import pickle

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFAR10Dataset(Dataset):
    def __init__(self, data_dir='../coding/Dataset/cifar-10-batches-py/', mode='train'):
        if mode == 'train':
            self.data = self.load_all_batches(data_dir, 'data_batch')
        elif mode == 'test':
            self.data = self.load_batch(data_dir, 'test_batch')
        else:
            raise ValueError("Mode doit être 'train' ou 'test'.")
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[125.3/255, 123.0/255, 113.9/255], std=[63.0/255, 62.1/255, 66.7/255])
        ])

    def __len__(self):
        return len(self.data['data'])

    def __getitem__(self, idx):
        img = Image.fromarray(self.data['data'][idx].reshape(3, 32, 32).transpose(1, 2, 0))
        img = self.transform(img)
        label = self.data['labels'][idx]
        return img, label

    def load_batch(self, data_dir, batch_name):
        batch_file = os.path.join(data_dir, batch_name)
        batch_data = self.unpickle(batch_file)
        return {'data': batch_data[b'data'], 'labels': batch_data[b'labels']}

    def load_all_batches(self, data_dir, base_name):
        all_data = {'data': [], 'labels': []}
        for batch_num in range(1, 6):  # Chargement des 5 batchs
            batch_file = os.path.join(data_dir, f"{base_name}_{batch_num}")
            batch_data = self.unpickle(batch_file)
            all_data['data'].extend(batch_data[b'data'])
            all_data['labels'].extend(batch_data[b'labels'])
        return all_data

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict



# class CIFAR100Dataset(Dataset):
#     def __init__(self, data_dir="../Dataset/CIFAR10/cifar-10-batches-py", mode='train'):
#         if mode == 'train':
#             self.data = self.load_batch(data_dir, 'train')
#         elif mode == 'test':
#             self.data = self.load_batch(data_dir, 'test')
#         else:
#             raise ValueError("Mode doit être 'train' ou 'test'.")

#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[129.3/255, 124.1/255, 112.4/255], std=[68.2/255, 65.4/255, 70.4/255])
#         ])

#     def __len__(self):
#         return len(self.data['data'])

#     def __getitem__(self, idx):
#         img = Image.fromarray(self.data['data'][idx].reshape(3, 32, 32).transpose(1, 2, 0))
#         img = self.transform(img)
#         label = self.data['labels'][idx]
#         return img, label

#     def load_batch(self, data_dir, batch_name):
#         batch_file = os.path.join(data_dir, batch_name)
#         batch_data = self.unpickle(batch_file)
#         return {'data': batch_data[b'data'], 'labels': batch_data[b'fine_labels']}

#     def unpickle(self, file):
#         with open(file, 'rb') as fo:
#             dict = pickle.load(fo, encoding='bytes')
#         return dict


if __name__=="__main__":

    test_dataset = CIFAR10Dataset(mode='train')

    # Exemple d'accès à un élément du jeu de données de test
    sample_img, sample_label = test_dataset[50]
    print("Image shape:", sample_img.shape)
    print("Label:", sample_label)

