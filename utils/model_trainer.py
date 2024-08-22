import copy
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.losses import DiceLoss, TverskyLoss
# from NASNet.network_generator import Network
from utils.pytorch_dataset import RadarDataset
# from unet import UNet


def make_image(x, y, probabilities):
    f, ax = plt.subplots(1, 3, figsize=(16,8))
    ax[0].imshow(x[0,0].detach().cpu().numpy())
    ax[1].imshow(y[0,0].detach().cpu().numpy())
    ax[2].imshow(probabilities[0,0].detach().cpu().numpy())
    plt.close(f)
    return f

class ModelTrainer:

    def __init__(self, model=None, train_data=None, validation_data=None, test_data=None,
                 optimizer=None, n_epochs=None, device=None, loss_fn=None,
                 save_path=None, random_seed=None, threshold=None, params_path=None, disable_tqdm=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.n_epochs = n_epochs
        self.loss_fn = loss_fn
        self.save_path = save_path
        self.random_seed = random_seed
        self.threshold = threshold
        self.params_path = params_path
        self.disable_tqdm = disable_tqdm
        self.scheduler = None

    def set_model(self, model):
        self.model = model

    def set_data(self, train_data, validation_data=None):
        self.train_data = train_data
        self.validation_data = validation_data

    def set_test_data(self, test_data):
        self.test_data = test_data

    def set_parameters(self, path):

        assert self.model is not None, "We need a model to initialize the optimizer"
        with open(path, "r") as f:
            dic = json.load(f)

        lr = dic["training"]["lr"]
        opt = dic["training"]["optimizer"]
        if opt == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(self.optimizer, gamma=0.9)
        self.n_epochs = dic["training"]["n_epochs"]

        loss = dic["training"]["loss"]
        if loss == "dice":
            self.loss_fn = DiceLoss()
        elif loss == "binary_cross_entropy":
            self.loss_fn = torch.nn.BCELoss()
        elif loss == "categorical_cross_entropy":
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss == "tversky":
            self.loss_fn = TverskyLoss(alpha=0.5)

        self.device = torch.device(dic["hardware"]["device"])
        self.model.to(self.device)

        self.save_path = dic["general"]["save_path"]
        # torch.manual_seed(dic["general"]["seed"])
        # np.random.seed(dic["general"]["seed"])
        # print(f"Random seed set to {dic['general']['seed']}")

        self.threshold = dic["general"]["threshold"]

    def train_model(self, log_dir=None):

        if log_dir is not None:
            writer = SummaryWriter(log_dir=log_dir)
            if isinstance(self.model, Network):
                writer.add_graph(self.model.encoder[0], [torch.zeros(1,1,256,256).to(self.device), torch.zeros(1,1,256,256).to(self.device)])
            elif isinstance(self.model, UNet):
                writer.add_graph(self.model, torch.zeros(1, 1, 256, 256).to(self.device))

        for i in range(self.n_epochs):

            # Train loop
            pbar = tqdm(self.train_data, disable=self.disable_tqdm)
            loss_train = []
            for step, (x, target) in enumerate(pbar):
                self.optimizer.zero_grad()

                x = x.to(self.device)
                target = target.to(self.device)

                probabilities = self.model.forward(x)
                loss = self.loss_fn(probabilities, target)
                tp = torch.sum((probabilities > self.threshold) * target)
                fp = torch.sum((probabilities > self.threshold) * (1 - target))
                tn = torch.sum((1 - (probabilities > self.threshold).float()) * (1 - target))
                fn = torch.sum((1 - (probabilities > self.threshold).float()) * target)
                recall = tp / (tp + fn)

                loss_train.append(loss.cpu().detach().numpy())
                loss.backward()

                if step % 10 == 0 and log_dir is not None:  # Write image, prediction and label
                    fig = make_image(x, target, probabilities)
                    writer.add_figure("prediction", fig, global_step=i*len(pbar)+step)
                    writer.add_histogram("output distribution", probabilities.view(-1), global_step=i*len(pbar)+step)

                self.optimizer.step()

                """
                Tensorboard metrics
                """
                if log_dir is not None:
                    writer.add_scalar("Loss/train_step", loss, i*len(pbar)+step)
                    writer.add_scalar("Recall/train_step", recall, i*len(pbar)+step)

                pbar.set_description(f"[{i}/{self.n_epochs}] : {np.round(np.mean(loss_train), 6)}")

            torch.cuda.empty_cache()

            # Eval loop
            pbar = tqdm(self.validation_data, disable=self.disable_tqdm)
            loss_eval = []
            for step, (x, target) in enumerate(pbar):
                x = x.to(self.device)
                target = target.to(self.device)
                probabilities = self.model.forward(x)
                loss = self.loss_fn(probabilities, target)
                tp = torch.sum((probabilities > self.threshold) * target)
                fp = torch.sum((probabilities > self.threshold) * (1 - target))
                tn = torch.sum((1 - (probabilities > self.threshold).float()) * (1 - target))
                fn = torch.sum((1 - (probabilities > self.threshold).float()) * target)
                recall = tp / (tp + fn)
                loss_eval.append(loss.cpu().detach().numpy())
                if log_dir is not None:
                    writer.add_scalar("Loss/val_step", loss, i*len(pbar)+step)
                    writer.add_scalar("Recall/val_step", recall, i*len(pbar)+step)
                pbar.set_description(f"[EVAL {i}/{self.n_epochs}] : {np.round(np.mean(loss_eval), 6)}")
            if log_dir is not None:
                writer.add_scalar("Loss/train", np.mean(loss_train), i)
                writer.add_scalar("Loss/val", np.mean(loss_eval), i)
            self.scheduler.step()

            if log_dir is not None:
                print(f"Saving weights at {self.save_path}/model_epoch-{i}.pt")
                _ = self.save_weights(f"{log_dir.split('/')[-1]}model_epoch-{i}.pt")
        if log_dir is not None: writer.flush()

    def save_weights(self, filename="model.pt"):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, filename))
        return filename

    def load_weights(self, filename):
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, filename)))

    def evaluate(self):
        assert self.test_data is not None

        true_positives = []
        true_negatives = []
        false_positives = []
        false_negatives = []
        accuracy = []
        losses = []

        for inputs_numpy, targets_numpy in tqdm(self.test_data, disable=self.disable_tqdm):
            inputs = inputs_numpy.to(self.device)
            targets = targets_numpy.to(self.device)

            outputs = self.model.forward(inputs)

            tp = torch.sum((outputs > self.threshold) * targets)
            fp = torch.sum((outputs > self.threshold) * (1 - targets))
            tn = torch.sum((1 - (outputs > self.threshold).float()) * (1 - targets))
            fn = torch.sum((1 - (outputs > self.threshold).float()) * targets)
            acc = (tp + tn) / (tp+fp+tn+fn)
            loss = self.loss_fn(outputs, targets)

            true_positives.append(tp.cpu().detach().numpy())
            false_positives.append(fp.cpu().detach().numpy())
            true_negatives.append(tn.cpu().detach().numpy())
            false_negatives.append(fn.cpu().detach().numpy())
            accuracy.append(acc.cpu().detach().numpy())
            losses.append(loss.cpu().detach().numpy())

        return {"true_positives": true_positives,
                "false_positives": false_positives,
                "true_negatives": true_negatives,
                "false_negatives": false_negatives,
                "accuracy": accuracy,
                "loss": losses}

class DistilledModelTrainer(ModelTrainer):

        def __init__(self, model=None, train_data=None, validation_data=None, test_data=None,
                     optimizer=None, n_epochs=None, device=None, loss_fn=None,
                     save_path=None, random_seed=None, threshold=None, params_path=None, disable_tqdm=False):
            super().__init__()
            self.unet = UNet(128)

        def set_parameters(self, path):
            super().set_parameters(path)
            self.unet.to(self.device)

        def train_model(self, log_dir=None):

            if log_dir is not None:
                writer = SummaryWriter(log_dir=log_dir)
                if isinstance(self.model, Network):
                    writer.add_graph(self.model.encoder[0], [torch.zeros(1, 1, 256, 256).to(self.device),
                                                             torch.zeros(1, 1, 256, 256).to(self.device)])
                elif isinstance(self.model, UNet):
                    writer.add_graph(self.model, torch.zeros(1, 1, 256, 256).to(self.device))

            for i in range(self.n_epochs):

                # Train loop
                pbar = tqdm(self.train_data, disable=self.disable_tqdm)
                loss_train = []
                for step, (x, target) in enumerate(pbar):
                    self.optimizer.zero_grad()

                    x = x.to(self.device)
                    target = target.to(self.device)

                    probabilities = self.model.forward(x)
                    unet_probabilities = self.unet.forward(x)
                    temperature = 20
                    unet_dist = torch.exp(unet_probabilities / temperature) / torch.sum(
                        torch.exp(unet_probabilities / temperature))
                    model_dist = torch.exp(probabilities / temperature) / torch.sum(
                        torch.exp(probabilities / temperature))

                    loss_example = self.loss_fn(probabilities, target)
                    loss_distillation = F.binary_cross_entropy_with_logits(model_dist.view(-1), unet_dist.view(-1))
                    loss = self.alpha * loss_example + (1 - self.alpha) * loss_distillation
                    tp = torch.sum((probabilities > self.threshold) * target)
                    fp = torch.sum((probabilities > self.threshold) * (1 - target))
                    tn = torch.sum((1 - (probabilities > self.threshold).float()) * (1 - target))
                    fn = torch.sum((1 - (probabilities > self.threshold).float()) * target)
                    recall = tp / (tp + fn)

                    loss_train.append(loss.cpu().detach().numpy())
                    loss.backward()

                    if step % 10 == 0 and log_dir is not None:  # Write image, prediction and label
                        fig = make_image(x, target, probabilities)
                        writer.add_figure("prediction", fig, global_step=i * len(pbar) + step)
                        writer.add_histogram("output distribution", probabilities.view(-1),
                                             global_step=i * len(pbar) + step)

                    self.optimizer.step()

                    """
                    Tensorboard metrics
                    """
                    if log_dir is not None:
                        writer.add_scalar("Loss/train_step", loss, i * len(pbar) + step)
                        writer.add_scalar("Recall/train_step", recall, i * len(pbar) + step)

                    pbar.set_description(f"[{i}/{self.n_epochs}] : {np.round(np.mean(loss_train), 6)}")

                torch.cuda.empty_cache()

                # Eval loop
                pbar = tqdm(self.validation_data, disable=self.disable_tqdm)
                loss_eval = []
                for step, (x, target) in enumerate(pbar):
                    x = x.to(self.device)
                    target = target.to(self.device)
                    probabilities = self.model.forward(x)
                    loss = self.loss_fn(probabilities, target)
                    tp = torch.sum((probabilities > self.threshold) * target)
                    fp = torch.sum((probabilities > self.threshold) * (1 - target))
                    tn = torch.sum((1 - (probabilities > self.threshold).float()) * (1 - target))
                    fn = torch.sum((1 - (probabilities > self.threshold).float()) * target)
                    recall = tp / (tp + fn)
                    loss_eval.append(loss.cpu().detach().numpy())
                    if log_dir is not None:
                        writer.add_scalar("Loss/val_step", loss, i * len(pbar) + step)
                        writer.add_scalar("Recall/val_step", recall, i * len(pbar) + step)
                    pbar.set_description(f"[EVAL {i}/{self.n_epochs}] : {np.round(np.mean(loss_eval), 6)}")
                if log_dir is not None:
                    writer.add_scalar("Loss/train", np.mean(loss_train), i)
                    writer.add_scalar("Loss/val", np.mean(loss_eval), i)
                self.scheduler.step()
                print(f"Saving weights at {self.save_path}/model_epoch-{i}.pt")
                _ = self.save_weights(f"{log_dir.split('/')[-1]}model_epoch-{i}.pt")
            if log_dir is not None: writer.flush()