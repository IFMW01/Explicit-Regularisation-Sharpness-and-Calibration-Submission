# %%
import pprint
from pathlib import Path
from pprint import pprint

import torch
import yaml
from easydict import EasyDict

from data_loader_manager import dataloaders
from models import vgg_model
from trainers.metrics_processor import MetricsProcessor
from models.temperature_scaling import ModelWithTemperature

with open("../configs/igs_config.yaml", "r") as file:
    config = yaml.safe_load(file)


pprint(config)

config = EasyDict(config)
config.aug = False
config.adversarial = False

config.sharpness_batch_size = 64
config.sharpness_dataset_size = 64*20

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

model = vgg_model.VGG(
    vgg_name="VGG19",
    dropout=0.0,
    vgg_config=Path(config.config_dir) / config.vgg_config,
    num_classes=10
)
model = model.to(device)

seed = 42

# %%
model_dir = Path("../models/CIFAR10/")

# %%
import glob

import numpy as np

for model_path in glob.glob(f"{str(model_dir)}/*/*/best_baseline*.pth"):
  print(model_path)
  model.load_state_dict(torch.load(model_path, map_location=device))
  data_loader_manager = dataloaders.DataLoaderManager(
      config=config,
      dataset_name="CIFAR10",
      seed=seed,
  )

  train_dataloader, dev_dataloader, test_dataloader, sharpness_dataloader = data_loader_manager.get_dataloaders()

  model.eval()
  mp = MetricsProcessor(
      config=config,
      model=model,
      train_dataloader=sharpness_dataloader,
      test_dataloader=sharpness_dataloader,
      model_name="",
      device=device,
      seed=seed,
      num_classes=data_loader_manager.num_classes
  )

  igs = np.log(mp.IGS(output_all=True))
  print(igs)
  print(igs.mean(), igs.std())

  with (model_dir / "igs_output").open("a") as logfile:
      logfile.write(f"{model_path},{igs.mean()},{igs.std()}\n")

# %%
model = ModelWithTemperature(model, device=device, temperature=1.0).to(device)

for model_path in glob.glob(f"{str(model_dir)}/*/*/best_with_temperature_baseline*.pth"):
  print(model_path)
  model.load_state_dict(torch.load(model_path, map_location=device))
  data_loader_manager = dataloaders.DataLoaderManager(
      config=config,
      dataset_name="CIFAR10",
      seed=seed,
  )

  train_dataloader, dev_dataloader, test_dataloader, sharpness_dataloader = data_loader_manager.get_dataloaders()

  model.eval()
  mp = MetricsProcessor(
      config=config,
      model=model,
      train_dataloader=sharpness_dataloader,
      test_dataloader=sharpness_dataloader,
      model_name="",
      device=device,
      seed=seed,
      num_classes=data_loader_manager.num_classes
  )

  igs = np.log(mp.IGS(output_all=True))
  print(igs)
  print(igs.mean(), igs.std())

  with (model_dir / "igs_output").open("a") as logfile:
      logfile.write(f"{model_path},{igs.mean()},{igs.std()}\n")

