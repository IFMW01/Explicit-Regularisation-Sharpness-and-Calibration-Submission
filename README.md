# R252 Theory of Deep Learning Project

## Setup

Conda environment:

```
conda create --prefix r252-env python=3.8
conda activate path/to/here/r252-env
```

Install PyTorch:

```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Install other requirements

```
pip install -r requirements.txt
```


## Runs

### Random initialization:
```
python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --baseline True \
    --dataset CIFAR10  \
    --save_name baseline 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --baseline True \
    --dropout 0.1 \
    --dataset CIFAR10  \
    --save_name baseline_dropout_0_1 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --baseline True \
    --aug True \
    --dataset CIFAR10  \
    --save_name baseline_augmentations 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --baseline True \
    --weight_decay 0.05 \
    --dataset CIFAR10  \
    --save_name baseline_weight_decay_0_05
```

### Adversarial iniialization

```
python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --baseline True \
    --adversarial True \
    --dataset CIFAR10  \
    --save_name adversarial_initialization 
```

```
python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --adversarial True \
    --dataset CIFAR10  \
    --save_name adversarial_baseline 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --adversarial True \
    --dropout 0.1 \
    --dataset CIFAR10  \
    --save_name adversarial_dropout_0_1 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --adversarial True \
    --aug True \
    --dataset CIFAR10  \
    --save_name adversarial_augmentations 

python main.py \
    --config_file ../configs/base_config.yaml \
    --model_name VGG19 \
    --adversarial True \
    --weight_decay 0.05 \
    --dataset CIFAR10  \
    --save_name adversarial_weight_decay_0_05
```



Repository from the Visualizing Loss Landscapes Paper to visualise loss landscape results (src/vizualisation)

```
@inproceedings{visualloss,
  title={Visualizing the Loss Landscape of Neural Nets},
  author={Li, Hao and Xu, Zheng and Taylor, Gavin and Studer, Christoph and Goldstein, Tom},
  booktitle={Neural Information Processing Systems},
  year={2018}
}
```
