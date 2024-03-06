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
        seed,
    ):
        self.config = config
        self.device = device
        self.model = model.to(self.device)
        self.num_classes = num_classes

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

        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("train/acc", summary="max")
        wandb.define_metric("train/ECE", summary="max")

        wandb.define_metric("eval/loss", summary="min")
        wandb.define_metric("eval/acc", summary="max")
        wandb.define_metric("eval/ECE", summary="max")

        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay = self.config.weight_decay
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

                train_accs = []

                for X, y in train_dataloader:
                    y = y.to(self.device)
                    X = X.to(self.device)

                    loss, pred = self.train_model(X, y, optimizer, loss_fn)

                    running_loss += loss

                    comparison_with_gold = torch.argmax(pred, dim=-1) == y
                    train_accs.append(np.mean(comparison_with_gold.detach().cpu().numpy()))

                    train_ece.update(pred, y)

                ece_test = MulticlassCalibrationError(
                    num_classes=self.num_classes, n_bins=15, norm="l1"
                ).to(self.device)
                running_test_loss = 0

                val_accs = []

                for X_val, y_val in eval_dataloader:
                    X_val = X_val.to(self.device)
                    y_val = y_val.to(self.device)

                    val_loss, val_pred = self.eval_model(X_val, y_val, loss_fn)

                    running_test_loss += val_loss
                    comparison_with_gold = torch.argmax(val_pred, dim=-1) == y_val
                    val_accs.append(np.mean(comparison_with_gold.detach().cpu().numpy()))
                    ece_test.update(val_pred, y_val)

                train_loss = running_loss / len(train_dataloader)
                train_ece = train_ece.compute().detach().cpu().item()
                val_loss = running_test_loss / len(eval_dataloader)
                val_ece = ece_test.compute().detach().cpu().item()

                tepoch.set_postfix(
                    train_loss=train_loss,
                    val_loss=val_loss,
                )

                train_acc = np.mean(train_accs)
                val_acc = np.mean(val_accs)

                wandb.log(
                    {
                        "train/loss": train_loss,
                        "train/ECE": train_ece,
                        "train/acc": train_acc,
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "eval/loss": val_loss,
                        "eval/ECE": val_ece,
                        "eval/acc": val_acc,
                    }
                )

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_model = copy.deepcopy(self.model)
                    best_epoch = epoch

        return best_model, self.model