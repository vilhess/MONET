import torch
import numpy as np
import os

from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

def preprocess_dava_card(card, mini, maxi, target_dim=128, label=False, random_crops=None, flip=False):
    # 1. Normalize card
    # if not label:
        # card = (card + mu) / sigma
        # card = (card - mini) / (maxi-mini)
    # 2. Crop card
    if random_crops is not None:
        card = card[:, random_crops[0]:card.shape[1]-random_crops[1], random_crops[2]:card.shape[2]-random_crops[3]]
    if flip:
        card = np.flip(card, axis=2)
    # 3. Zero pad card
    x_pad = target_dim - card.shape[1]
    y_pad = target_dim - card.shape[2]
    x_pad_1 = x_pad // 2
    x_pad_2 = x_pad - x_pad_1
    y_pad_1 = y_pad // 2
    y_pad_2 = y_pad - y_pad_1
    card = np.pad(card, ((0,0), (x_pad_1,x_pad_2), (y_pad_1,y_pad_2)), 'constant')
    return card


class RadarDataset(Dataset):

    def __init__(self, root_dir, max_idx=None, batch_size=8):
        self.root_dir = root_dir
        self.idx = list(range(len(os.listdir(self.root_dir)) // 2))
        self.batch_size = batch_size
        if max_idx is not None:
            self.idx = list(range(max_idx))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"carte{idx}.npy")
        label_name = os.path.join(self.root_dir, f"label{idx}.npy")
        x = torch.from_numpy(np.expand_dims(np.load(img_name), 0)).float()
        y = torch.from_numpy(np.expand_dims(np.load(label_name), 0)).float()
        return x, y

    def generate_loaders(self, test_split=0.8, val_split=0.8):
        train, test = torch.utils.data.random_split(self, [test_split, 1-test_split])
        train, val = torch.utils.data.random_split(train, [val_split, 1-val_split])
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        return train_loader, val_loader, test_loader

class RadarDavaDataset(Dataset):

    def __init__(self, root_dir, max_idx=None, batch_size=64, has_distance=False):
        self.root_dir = root_dir
        divider = 2 if not has_distance else 3
        self.idx = list(range(1, (len(os.listdir(self.root_dir)) // divider) - 1))
        if max_idx is not None:
            self.idx = list(range(max_idx))
        self.mini = -79
        self.maxi = 55

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.batch_size=batch_size

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"carte{idx}.mat")
        label_name = os.path.join(self.root_dir, f"label{idx}.mat")
        x = loadmat(img_name)["signal_db"]
        y = loadmat(label_name)["label"]
        random_crops = (np.random.randint(0, 15), np.random.randint(0, 15), np.random.randint(0, 25), np.random.randint(0, 25))
        flip = np.random.choice([True, False])
        n = int(idx / 8000)  # Va changer les stats de normalisation
        x_dava = preprocess_dava_card(x, self.mini, self.maxi, random_crops=random_crops, flip=flip)
        y_dava = preprocess_dava_card(y, self.mini, self.maxi, label=True, random_crops=random_crops, flip=flip)
        x = torch.from_numpy(x_dava).float()
        y = torch.from_numpy(y_dava).float()
        return x, y

    def generate_loaders(self, test_split=0.8, val_split=0.8):
        train, test = torch.utils.data.random_split(self, [test_split, 1-test_split])
        train, val = torch.utils.data.random_split(train, [val_split, 1-val_split])
        train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        return train_loader, val_loader, test_loader

class RadarDavaTestDataset(RadarDavaDataset):

    def __init__(self, root_dir, mini=-79, maxi=55, max_idx=None):
        super().__init__(root_dir, mini, maxi, max_idx)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, f"carte{idx}.mat")
        label_name = os.path.join(self.root_dir, f"label{idx}.mat")
        x = loadmat(img_name)["signal_db"]
        y = loadmat(label_name)["label"]
        random_crops = (np.random.randint(0, 15), np.random.randint(0, 15), np.random.randint(0, 25), np.random.randint(0, 25))
        flip = np.random.choice([True, False])
        n = int(idx / 80)  # Va changer les stats de normalisation
        x_dava = preprocess_dava_card(x, self.mus[n], self.sigmas[n], random_crops=random_crops, flip=flip)
        y_dava = preprocess_dava_card(y, self.mus[n], self.sigmas[n], label=True, random_crops=random_crops, flip=flip)
        x = torch.from_numpy(x_dava).float()
        y = torch.from_numpy(y_dava).float()
        return x, y

if __name__ == "__main__":
    PATH = "/projets/users/T0259728/radarconf24/mat"
    dataset = RadarDavaDataset(root_dir=PATH)
    x, y = dataset.__getitem__(1)
    print(f"x shape: {x.shape}, y_shape: {y.shape}")
    # plt.imshow(x.detach().numpy().squeeze())
    plt.hist(x.detach().numpy().squeeze().ravel(), bins=32)
    plt.show()

