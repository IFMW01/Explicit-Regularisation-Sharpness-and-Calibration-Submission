import argparse
import copy
import os
import random
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import wandb
import yaml
from easydict import EasyDict

import data_loader_manager.dataloaders as dataloaders
import models.vgg_model as vgg_model
import trainers.classification_executor as classification_executor
import trainers.metrics_processor as metrics_processor
from models.temperature_scaling import ModelWithTemperature


def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument(
        "--baseline",
        default=False,
        type=bool,
        help="Is this the baseline model: True is yes, LEAVE BLANK FOR NO.",
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
        help="Amount of dropout to be applied e.g. 0.05,0.1,0.15,0.5",
    )
    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Use 5e-4??" # values between 0 and 0.1 i.e. try 0.01, 0.05 and 0.1 and then take the best reult to present
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"Found {torch.cuda.device_count()} CUDA devices")
        print(f"({torch.cuda.get_device_name(device)})")
        return device
    else:
        print("No CUDA devices found, using CPU")
        return "cpu"


def save_model(model, save_file_name, save_dir):

    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    full_save_dir = save_dir / f"{save_file_name}.pth"
    torch.save(
        model.state_dict(),
        full_save_dir,
    )
    print("-----------------")
    print(f"Model saved at: {full_save_dir}")
    print("-----------------")


def main(seed=None, run_num=0):
    args = options_parser()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)

    config.update(vars(args))

    pprint(config)

    config = EasyDict(config)

    config.models_dir = Path(config.models_dir) / args.dataset

    config.models_dir = (
        config.models_dir / args.save_name / str(seed)
    )

    config.config_dir = Path(config.config_dir)

    config.seed = seed if seed is not None else config.seed

    set_seed(seed)
    device = get_device()

    print("Loading model")

    data_loader_manager = dataloaders.DataLoaderManager(
        config=config,
        dataset_name=args.dataset,
        seed=seed,
    )

    print("Loading dataloaders")

    (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        sharpness_train_dataloader,
    ) = data_loader_manager.get_dataloaders()

    if config.model_name.startswith("VGG"):
        model = vgg_model.VGG(
            vgg_name=args.model_name,
            dropout=args.dropout,
            num_classes=data_loader_manager.num_classes,
            vgg_config=config.config_dir / config.vgg_config,
        ).to(device)
    else:
        raise NotImplementedError(f"Model {config.model_name} not implemented")

    if args.baseline:
        approach = "baseline"
    else:
        raise Exception("Approach not specified")

    if args.aug:
        reg_method = "augmentation"
    elif args.dropout > 0.0:
        reg_method = "dropout_" + str(args.dropout)
    elif args.weight_decay > 0.0:
        reg_method = "weight_decay" + str(args.weight_decay)
    else:
        reg_method = "no-regularisation"

    if config.wandb.use_wandb:
        wandb.init(
            project=config.wandb.project,
            entity=config.wandb.entity,
            config=config,
            group=f"{args.model_name}-{args.dataset}-{approach}-with-{reg_method}",
        )
        wandb.run.name = (f"{args.model_name}-{args.dataset}-{approach}-with-{reg_method}-seed-{seed}"
        )

    if not os.path.isfile(f"{config.models_dir}/best_{config.save_name}.pth"):
        print(
            f"Loading classification executor with model from: {config.models_dir}/best_{config.save_name}.pth"
        )

        trainer = classification_executor.Executor(
            config=config,
            model=model,
            device=device,
            num_classes=data_loader_manager.num_classes,
            seed=seed,
        )
        if args.baseline:
            print("Random initialisation")
            save_model(model, save_file_name="initialisation", save_dir=config.models_dir)
        else:
            raise Exception("Approach not specified")

        model, best_model = trainer.train_eval_loop(train_dataloader, dev_dataloader)

        save_model(model, config.save_name, config.models_dir)
        save_model(best_model, f"best_{config.save_name}", config.models_dir)
    else:
        print(f"Found model in {config.models_dir}/best_{config.save_name}, loading it up!")
        best_model = copy.deepcopy(model)
        model.load_state_dict(
            torch.load(f"{config.models_dir}/{config.save_name}.pth", map_location=torch.device(device)), 
        )
        best_model.load_state_dict(
            torch.load(f"{config.models_dir}/best_{config.save_name}.pth", map_location=torch.device(device)),
        )


    if not os.path.isfile(
        f"{config.models_dir}/best_with_temperature_{config.save_name}.pth"
    ):

        print("Learning optimum temperature T value...")
        temp_model = ModelWithTemperature(model, device=device)
        temp_model.set_temperature(dev_dataloader)
        save_model(temp_model, f"with_temperature_{config.save_name}", config.models_dir)
        print("Without early stopping, T = ", temp_model.temperature.item())

        temp_best_model = ModelWithTemperature(best_model, device=device)
        temp_best_model.set_temperature(dev_dataloader)
        save_model(temp_best_model, f"best_with_temperature_{config.save_name}", config.models_dir)

        print("With early stopping, T = ", temp_best_model.temperature.item())
    else:
        temp_model = ModelWithTemperature(model, device=device)
        temp_model.load_state_dict(
            torch.load(f"{config.models_dir}/with_temperature_{config.save_name}.pth", map_location=torch.device(device)),
        )


        temp_best_model = ModelWithTemperature(model, device=device)
        temp_best_model.load_state_dict(
            torch.load(f"{config.models_dir}/best_with_temperature_{config.save_name}.pth", map_location=torch.device(device)),
        )

    model_dict = {
        "temp_model": temp_model.to(device),
        "temp_best_model": temp_best_model.to(device),
        "model": ModelWithTemperature(model, device=device, temperature=1.0).to(device),
        "best_model": ModelWithTemperature(best_model, device=device, temperature=1.0).to(device),
    }

    for model_name, torch_model in model_dict.items():
        print("Computing sharpness metrics for:", model_name)

        mp = metrics_processor.MetricsProcessor(
            config=config,
            model=torch_model,
            train_dataloader=sharpness_train_dataloader,
            test_dataloader=test_dataloader,
            device=device,
            seed=seed,
            model_name=model_name,
            num_classes=data_loader_manager.num_classes
        )

        results = mp.compute_metrics()
        if config.wandb.use_wandb:

            wandb.log(results)
    if config.wandb.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    args = options_parser()

    with open(args.config_file, "r") as file:
        config = yaml.safe_load(file)
    config.update(vars(args))
    seeds = config["seed"]
    # seeds = [43, 91, 17]
    for seed in seeds:
        main(seed)
