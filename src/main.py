import argparse
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict

import data_loader_manager.dataloaders as dataloaders
import models.vgg_model as vgg_model
import trainers.executor as executor
import trainers.metrics_processor as metrics_processor
import wandb


def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument(
        "--baseline",
        default=False,
        type=bool,
        help="Is this the baseline model: True is yes, LEAVE BLANK FOR NO.",
    )
    parser.add_argument(
        "--adversarial",
        default=False,
        type=bool,
        help="Is this the adversarial model: True is yes, LEAVE BLANK FOR NO.",
    )
    parser.add_argument(
        "--model_name",
        required=True,
        type=str,
        help='Type of Model: i.e. "VGG9","VGG16" or "VGG19"',
    )
    parser.add_argument(
        "--aug",
        default=False,
        type=bool,
        help="Are you using augmentation: True is yes, LEAVE BLANK FOR NO",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="Amount of dropout to be applied e.g. 0.05,0.1,0.15",
    )
    parser.add_argument(
        "--dataset", required=True, type=str, help="CIFAR10 or CIFAR100"
    )
    parser.add_argument(
        "--save_name",
        required=True,
        type=str,
        help="Name of the model you are creating ending in .pth",
    )
    parser.add_argument(
        "--config_file",
        required=True,
        type=str,
        help="Path to config file",
    )

    args = parser.parse_args()

    return args


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"


def main(seed=None, run_num=0):
    args = options_parser()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    config.update(vars(args))

    pprint(config)

    config = EasyDict(config)

    config.models_dir = Path(config.models_dir)
    if args.adversarial:
        config.models_dir = config.models_dir / "adversarial"
    else:
        config.models_dir = config.models_dir / "baseline"
    config.models_dir = (
        config.models_dir / args.dataset / args.model_name / args.save_name / str(seed)
    )

    config.config_dir = Path(config.config_dir)

    seed = seed if seed is not None else config.seed

    set_seeds(seed)
    device = get_device()

    if config.model_name.startswith("VGG"):
        model = vgg_model.VGG(
            vgg_name=args.model_name,
            dropout=args.dropout,
            vgg_config=config.config_dir / config.vgg_config,
        ).to(device)
    else:
        raise NotImplementedError(f"Model {config.model_name} not implemented")

    data_loader_manager = dataloaders.DataLoaderManager(
        config=config,
        dataset_name=args.dataset,
        seed=seed,
    )

    trainer = executor.Executor(
        config=config,
        model=model,
        device=device,
        num_classes=data_loader_manager.num_classes,
        save_name=args.save_name,
        seed=seed,
    )

    train_dataloader, eval_dataloader, sharpness_dataloader = data_loader_manager.get_dataloaders()

    if args.baseline:
        print("Random initialisation")
        trainer.save_model(model, save_file_name="initialisation")
    elif args.adversarial:
        print("Adversarial initialisation")
        adversarial_initialization = f"../models/adversarial/CIFAR10/VGG19/adversarial_initialization/{seed}/adversarial_initialization.pth"
        model.load_state_dict(
            torch.load(adversarial_initialization),
        )
    else:
        raise Exception("Approach not specified")

    if args.baseline and args.adversarial:
        approach = "baseline-adversarial"
    elif args.baseline:
        approach = "baseline"
    elif args.adversarial:
        approach = "from-adversarial"
    else:
        raise Exception("Approach not specified")

    if args.aug:
        reg_method = "augmentation"
    elif args.dropout > 0.0:
        reg_method = "dropout"
    else:
        reg_method = "no-regularisation"

    wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=config,
        group=config.wandb.experiment_name
        + f"-{args.model_name}-{args.dataset}-{approach}-with-{reg_method}",
    )
    wandb.run.name = (
        config["wandb"]["experiment_name"]
        + f"-{args.model_name}-{args.dataset}-{approach}-with-{reg_method}-seed-{seed}"
    )

    wandb.define_metric("train/loss", summary="min")
    wandb.define_metric("train/acc", summary="max")
    wandb.define_metric("train/ECE", summary="max")

    wandb.define_metric("eval/loss", summary="min")
    wandb.define_metric("eval/acc", summary="max")
    wandb.define_metric("eval/ECE", summary="max")

    trainer.train_eval_loop(train_dataloader, eval_dataloader)

    mp = metrics_processor.MetricsProcessor(
        config=config,
        model=trainer.model,
        dataloader=sharpness_dataloader,
        device=device,
        seed=seed,
    )

    results = mp.compute_metrics()

    wandb.log(results)

    wandb.finish()


if __name__ == "__main__":
    seeds = [43, 91, 17]
    for seed in seeds:
        main(seed)
