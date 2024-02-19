import torch
import numpy as np
import os
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.transforms import v2
from torchmetrics.classification import MulticlassCalibrationError
from torch.optim import SGD

import cifar10_random_lables as CIFAR10random
import cifar10_random_lables as CIFAR100random
import VGG 
import VGG_dropout

class CreateModels:
    def __init__(self,baseline: bool, adversarial:bool, model_type:str , aug:bool ,dropout: float, early_stopping:bool ,dataset:str,save_name:str,save_directory:str):
        self.model_type = model_type #"VGG9", "VGG16", "VGG19"
        self.baseline = baseline
        self.adversarial = adversarial 
        self.aug = aug 
        self.dropout = dropout 
        self.early_stopping  = early_stopping 
        self.dataset = dataset # "CIFAR100" 
        self.batch_size = 256
        self.epochs = 20
        self.patience = 10
        self.save_name = save_name # Path to save model
        self.save_directory= save_directory # Path to directory to save

    if __name__ == "__main__":
      main()

    def main(self):
        self.set_seeds()
        device = self.device()
        model = self.VGG()
        if self.baseline ==True:
            if self.adversarial == True:
                self.save_innit(model, 'initialisation_adversarial.pth')
                print('Initialisation saved, see: ' +str(self.save_directory)+ '/initialisation_adversarial.pth')
                self.epochs = 40
            else:
                self.save_innit(model, 'initialisation_base.pth')
                print('Initialisation saved, see: ' +str(self.save_directory)+ '/initialisation_base.pth')
        else:
            if self.adversarial == True:
                baseline_path = './'+self.save_directory + '/initialisation_adversarial.pth'
                self.epochs = 40
            else:
                baseline_path = './'+self.save_directory + '/initialisation_base.pth'
            model.load_state_dict(torch.load(baseline_path))

        loss_fn = nn.CrossEntropyLoss()
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)

        trainloader,testloader = self.dataloaders()
        if self.early_stopping == True:
            self.train_early_stopping(model,loss_fn,optimizer,trainloader,testloader,device)
            
        else:
            self.train(model,loss_fn,optimizer,trainloader,testloader,device)


    def set_seeds(self,seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)

    
    def base_transformations(self):
        transform = transforms.Compose(
        [transforms.ToTensor()])
        return transform
    
    
    def aug_transformations(self):
        if self.dataset == "CIFAR10":
            transform_aug = v2.Compose(
                [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor()])
            return transform_aug
        elif self.dataset =='CIFAR100':
            transform_aug = v2.Compose(
                [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR100),
                transforms.ToTensor()])
            return transform_aug
        
    
    def dataloaders(self):

        if self.adversarial == False:
            if self.aug == True:
                transformations = self.aug_transformations()
            else:
                transformations = self.base_transformations()

        if self.dataset == "CIFAR10":
            testset = datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transformations)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=2)
    
            # For adversarial we need a different training loader
            if self.adversarial == True:
                randomtrainset = CIFAR10random(1.0,root='./data',download=True,transform=transformations,train=True)
                trainloader = torch.utils.data.DataLoader(randomtrainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=2)
            else:
                trainset = datasets.CIFAR10(root='./data', train=True,
                                                        download=True, transform=transformations)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)
            return trainloader,testloader
        
        elif self.dataset == "CIFAR100":
            testset = datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transformations)
            testloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                         shuffle=False, num_workers=2)
            
            # For adversarial we need a different training loader
            if self.adversarial == True:
                randomtrainset = CIFAR100random(1.0,root='./data',download=True,transform=transformations,train=True)
                trainloader = torch.utils.data.DataLoader(randomtrainset, batch_size=self.batch_size,
                                            shuffle=True, num_workers=2)
            else:
                trainset = datasets.CIFAR100(root='./data', train=True,
                                                        download=True, transform=transformations)
                trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                        shuffle=True, num_workers=2)
            return trainloader,testloader
        
        
    def save_model(self,model):
        if os.path.isdir(self.save_directory) ==False:
            os.mkdir(self.save_directory)
        torch.save(model.state_dict(),str(self.save_directory+'/'+ self.save_name))
    
    def save_innit(self,model,path):
        if os.path.isdir(self.save_directory) ==False:
            os.mkdir(self.save_directory)
        torch.save(model.state_dict(),str(self.save_directory+'/'+ path))

    
    def device(self):
        if torch.cuda.is_available():
            print("Found", torch.cuda.device_count(), "CUDA devices!")
            device = torch.cuda.current_device()
            print("\tAttached device is", torch.cuda.get_device_name(device))
            return device
        else:
            print("We couldn't find any CUDA devices attached to this session!")

    
    def VGG(self):
        if self.dropout > 0:
            return VGG_dropout.VGGDropout(str(self.model_type),self.dropout)
        else:
            return VGG.VGG(str(self.model_type))

    # TRAINING 
    def save_model(self,model):
        if os.path.isdir(self.save_directory) ==False:
            os.mkdir(self.save_directory)
        torch.save(model.state_dict(),str(self.save_directory+'/'+ self.save_name))


    def evaluate_model(self,model,testloader,loss_fn,device,epoch=None):
        size_test = int(np.ceil(len(testloader.dataset)//testloader.batch_size))
        model.to(device)
        model.eval() # Stop any weight updates on the model (i.e. Batch weights)
        running_test_loss = 0
        ece_test = MulticlassCalibrationError(num_classes=10,n_bins=15,norm='l1').to(device)
        for test_batch, (X_val, y_val) in enumerate(testloader):
            x_val = X_val.to(device)
            y_val = y_val.to(device)
            with torch.no_grad():
                val_pred = model(x_val)
                val_loss = loss_fn(val_pred, y_val)
                running_test_loss += val_loss.item()
                val_acc = np.mean(
                    (torch.argmax(val_pred, dim=-1) == y_val).detach().cpu().numpy()
                )
                ece_test.update(val_pred,y_val)
        print(
            "-"*10, "TEST ACC", "-"*10,
            f"val loss: {running_test_loss/len(testloader):>5f}, val accuracy: {val_acc:.4f} "
            f"Test ECE: {ece_test.compute().detach().cpu().item():>5f} "
            )
        if epoch!= None:
            print(f"[epoch {epoch} and batch {test_batch}/{size_test}]",
            "-"*10,"TEST ACC","-"*10)

    def train(self,model,loss_fn,optimizer,trainloader,testloader,device):
        size_train = int(np.ceil(len(trainloader.dataset)//trainloader.batch_size))
        model.to(device)
        step = 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            ece_train = MulticlassCalibrationError(num_classes=10,n_bins=15,norm='l1').to(device)
            running_loss = 0
            for batch, (X, y) in enumerate(trainloader):
                optimizer.zero_grad()
                X = X.to(device)
                y = y.to(device)
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                acc = np.mean(
                    (torch.argmax(pred, dim=-1) == y).detach().cpu().numpy()
                )
                ece_train.update(pred,y)
            print(
                f"loss: {running_loss/len(trainloader):>7f}, train accuracy: {acc:.5f} "
                f"Train ECE: {ece_train.compute().detach().cpu().item():>5f} "
                f"[epoch {epoch} and batch {batch}/{size_train} (step {step})]"
            )
            if epoch % 10==0:
                self.evaluate_model(model,testloader,loss_fn,device,epoch=epoch)

        print('Finished Training')
        print('Model saved, see: ' +str(self.save_directory+'/'+ self.save_name))
        self.save_model(model)

    def train_early_stopping(self,model,loss_fn,optimizer,trainloader,testloader,device):
        size_train = int(np.ceil(len(trainloader.dataset)//trainloader.batch_size))
        size_test = int(np.ceil(len(testloader.dataset)//testloader.batch_size))
        model.to(device)
        step = 0
        best_acc = 0
        for epoch in range(1, self.epochs + 1):
            model.train()
            ece_train = MulticlassCalibrationError(num_classes=10,n_bins=15,norm='l1').to(device)
            running_loss = 0
            running_test_loss = 0
            for batch, (X, y) in enumerate(trainloader):
                optimizer.zero_grad()
                X = X.to(device)
                y = y.to(device)
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)
                # Backpropagation
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                acc = np.mean(
                    (torch.argmax(pred, dim=-1) == y).detach().cpu().numpy()
                )
                ece_train.update(pred,y)
            print(
                f"loss: {running_loss/len(trainloader):>7f}, train accuracy: {acc:.5f} "
                f"Train ECE: {ece_train.compute().detach().cpu().item():>5f} "
                f"[epoch {epoch} and batch {batch}/{size_train} (step {step})]"
            )

            model.eval() # Stop any weight updates on the model (i.e. Batch weights)
            ece_test = MulticlassCalibrationError(num_classes=10,n_bins=15,norm='l1').to(device)
            for test_batch, (X_val, y_val) in enumerate(testloader):
                x_val = X_val.to(device)
                y_val = y_val.to(device)
                with torch.no_grad():
                    val_pred = model(x_val)
                    val_loss = loss_fn(val_pred, y_val)
                    running_test_loss += val_loss.item()
                    val_acc = np.mean(
                        (torch.argmax(val_pred, dim=-1) == y_val).detach().cpu().numpy()
                    )
                    ece_test.update(val_pred,y_val)
            print(
                "-"*10, "TEST ACC", "-"*10,
                f"val loss: {running_test_loss/len(testloader):>5f}, val accuracy: {val_acc:.4f} "
                f"Test ECE: {ece_test.compute().detach().cpu().item():>5f} "
                f"[epoch {epoch} and batch {batch}/{size_test}]",
                "-"*10,"TEST ACC","-"*10,
                )
            if val_acc >= best_acc:
                best_acc = val_acc
                best_model = model
                consecutive_no_improvement = 0
            else:
                consecutive_no_improvement += 1
                if consecutive_no_improvement >= self.patience:
                    print(f'Early stopping after {self.patience} consecutive epochs without improvement.')
                    break
        print('Finished Training')
        self.save_model(best_model)
        print('Best model saved, see: ' +str(self.save_directory+'/'+ self.save_name))

    





    
    
