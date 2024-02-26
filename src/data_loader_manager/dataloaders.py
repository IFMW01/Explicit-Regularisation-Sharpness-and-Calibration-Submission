import copy
import os
import random

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import data_loader_manager.cifar_random_lables as cifar_random_lables


class DataLoaderManager:
    def __init__(
        self,
        config,
        dataset_name: str,
        seed: int,
    ):
        self.config = config

        self.dataset_name = dataset_name
        if dataset_name == "CIFAR10":
            self.dataset = datasets.CIFAR10
            self.num_classes = 10
            self.policy = transforms.AutoAugmentPolicy.CIFAR10
        elif dataset_name == "CIFAR100":
            self.dataset = datasets.CIFAR100
            self.num_classes = 100
            self.policy = transforms.AutoAugmentPolicy.CIFAR10
        else:
            raise ValueError("Only CIFAR10 and CIFAR100 supported")

        self.seed = seed

    def get_dataloaders(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        if self.config.aug == True:
            transformations = self.aug_transformations()
        else:
            transformations = self.base_transformations()

        eval_dataset = self.dataset(
            root=self.config.cifar_dir,
            train=False,
            download=True,
            transform=transformations,
        )

        if self.config.adversarial == True and self.config.baseline == True:
            print("Loading random label train data")
            train_dataset = cifar_random_lables.get_random_cifar_dataset(
                self.dataset,
                self.num_classes,
                corrupt_prob=1.0,
                root=self.config.cifar_dir,
                download=True,
                transform=transformations,
                train=True,
            )
        else:
            print("Loading normal train data")
            train_dataset = self.dataset(
                root=self.config.cifar_dir,
                train=True,
                download=True,
                transform=transformations,
            )

        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=self.seed_worker,
            generator=g,
        )

        if self.config.sharpness_dataset_size == -1:
            sharpness_dataset = train_dataset
        else:
            sharpness_dataset = torch.utils.data.Subset(
                train_dataset, list(range(0, self.config.sharpness_dataset_size)))

        if self.config.sharpness_batch_size == -1:
            print(len(sharpness_dataset))
            sharpness_batch_size = len(sharpness_dataset)
        else:
            sharpness_batch_size = self.config.sharpness_batch_size

        sharpness_dataloader = torch.utils.data.DataLoader(
            sharpness_dataset,
            batch_size=sharpness_batch_size,
            shuffle=True,
            num_workers=2,
            worker_init_fn=self.seed_worker,
            generator=g,
        )
        return train_dataloader, eval_dataloader, sharpness_dataloader

    def base_transformations(self):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def aug_transformations(self):

        transform_aug = transforms.Compose(
            [
                transforms.AutoAugment(self.policy),
                transforms.ToTensor(),
            ]
        )
        return transform_aug
