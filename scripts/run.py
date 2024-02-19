import create_models as CM
import argparse
import torch
import torch.nn as nn
from torch.optim import SGD


def options_parser():
    parser = argparse.ArgumentParser(description="Arguments for creating model")
    parser.add_argument('--baseline',default=True, type = bool, help='Is this the baseline model: True is yes, False is no.')
    parser.add_argument('--adversarial',default=False, type = bool, help='Is this the adversarial model: True is yes, False is no.')
    parser.add_argument('--model_type',default='VGG19',required=True, type = str, help='Type of Model: i.e. "VGG9","VGG16" or "VGG19"')
    parser.add_argument('--aug',default=False,type = bool, help='Are you using augmentation: True is yes, False is no')
    parser.add_argument('--dropout',default=0.0, type = float, help='Amount of dropout to be applied: 0.05,0.1,0.15')
    parser.add_argument('--early_stopping',default=False, required=True, type = bool, help='Are you using early stopping: True is yes, False is no')
    parser.add_argument('--dataset',required=True, type = str, help='CIFAR10 or CIFAR100')
    parser.add_argument('--save_name',required=True, type = str, help='Name of the model you are creating ending in .pth')
    parser.add_argument('--save_directory',required=True, type = str, help='Name of directory to save the model to')

    args = parser.parse_args()

    return args


def main():
    args = options_parser()
    model_created = CM(args.baseline,args.adversarial,args.model_type,args.aug,args.dropout,args.dropout_amount,args.early_stopping,args.dataset,args.save_name,args.save_directory)
    model_created.set_seeds()
    device = model_created.device()
    model = model_created.VGG()
    if model_created.baseline ==True:
        if model_created.adversarial == True:
            model_created.save(model, 'initialisation_adversarial.pth')
            model_created.epochs = 40
        else:
            model_created.save_model(model, 'initailisation.pth')
    else:
        if model_created.adversarial == True:
            baseline_path = model_created.save_directory + '/initailisation.pth'
        else:
            baseline_path = model_created.save_directory + '/initailisation.pth'
        model.load_state_dict(torch.load(baseline_path))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

    trainloader,testloader = model_created.dataloaders()
    if model_created.early_stopping == True:
        model_created.train_early_stopping(model,loss_fn,optimizer,trainloader,testloader,device)
        
    else:
        model_created.train(model,loss_fn,optimizer,trainloader,testloader,device)


if __name__ == "__main__":
    main()