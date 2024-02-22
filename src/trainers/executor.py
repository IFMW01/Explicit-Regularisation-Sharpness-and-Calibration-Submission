import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import SGD
from torchmetrics.classification import MulticlassCalibrationError
from tqdm import tqdm

import wandb


class Executor:
    def __init__(
        self,
        config,
        model,
        device,
        num_classes,
        save_name,
        seed,
    ):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.num_classes = num_classes

        self.save_name = save_name

        self.seed = seed

    def train_model(self, X, y, optimizer, loss_fn):
        self.model.train()
        optimizer.zero_grad()
        pred = self.model(X).to(self.device)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        return loss.item(), pred

    @torch.no_grad()
    def eval_model(self, X_val, y_val, loss_fn):
        self.model.eval()
        val_pred = self.model(X_val)
        val_loss = loss_fn(val_pred, y_val)
        return val_loss.item(), val_pred

    def train_eval_loop(self, train_dataloader, eval_dataloader):

        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
        )

        best_acc = 0
        best_epoch = 0

        with tqdm(range(1, self.config.num_epochs + 1), unit="e") as tepoch:
            for epoch in tepoch:

                tepoch.set_description(f"Epoch {epoch-1}")

                train_ece = MulticlassCalibrationError(
                    num_classes=self.num_classes, n_bins=15, norm="l1"
                ).to(self.device)
                running_loss = 0

                for X, y in train_dataloader:
                    y = y.to(self.device)
                    X = X.to(self.device)

                    loss, pred = self.train_model(X, y, optimizer, loss_fn)

                    running_loss += loss

                    comparison_with_gold = torch.argmax(pred, dim=-1) == y
                    train_acc = np.mean(comparison_with_gold.detach().cpu().numpy())

                    train_ece.update(pred, y)

                ece_test = MulticlassCalibrationError(
                    num_classes=self.num_classes, n_bins=15, norm="l1"
                ).to(self.device)
                running_test_loss = 0

                for X_val, y_val in eval_dataloader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)

                    val_loss, val_pred = self.eval_model(X_val, y_val, loss_fn)

                    running_test_loss += val_loss
                    comparison_with_gold = torch.argmax(val_pred, dim=-1) == y_val
                    val_acc = np.mean(comparison_with_gold.detach().cpu().numpy())
                    ece_test.update(val_pred, y_val)

                train_loss = running_loss / len(train_dataloader)
                train_ece = train_ece.compute().detach().cpu().item()
                val_loss = running_test_loss / len(eval_dataloader)
                val_ece = ece_test.compute().detach().cpu().item()

                tepoch.set_postfix(
                    train_loss=train_loss,
                    # train_ece=train_ece,
                    val_loss=val_loss,
                    # val_ece=val_ece,
                    # val_acc=val_acc,
                )

                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/ECE": train_ece,
                        "train/acc": train_acc,
                        "eval/loss": val_loss,
                        "eval/ECE": val_ece,
                        "eval/acc": val_acc,
                    }
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(self.model)
                    best_epoch = epoch

        print("-----------------")
        self.save_model(best_model, file_prefix="best")
        print(f"Best model at epoch {best_epoch} with val acc: {best_acc}")
        print(f"Saved at: {self.config.models_dir}/best_{self.save_name}")

        print("-----------------")
        self.save_model(self.model)
        print(f"Last model (epoch {self.config.num_epochs}) with val acc: {val_acc}")
        print(f"Saved at: {self.config.models_dir}/best_{self.save_name}")
        print("-----------------")

    def save_model(self, model, save_file_name=None, file_prefix=""):

        if save_file_name is None:
            save_file_name = self.save_name

        if file_prefix:
            file_prefix = f"{file_prefix}_"
        if os.path.isdir(self.config.models_dir) == False:
            os.makedirs(self.config.models_dir)

        full_save_dir = self.config.models_dir / f"{file_prefix}{save_file_name}.pth"
        torch.save(
            model.state_dict(),
            full_save_dir,
        )
        print(f"Model saved at: {full_save_dir}")
