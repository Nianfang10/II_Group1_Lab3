import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam, lr_scheduler

import torchvision
import torchvision.transforms as transforms


from tqdm import tqdm

from dataset import Dataset
from networks import RNN,LSTM,GRU
import pdb
import wandb
from datetime import datetime



class UpdatingMean():
    def __init__(self) -> None:
        self.sum = 0
        self.n = 0

    def mean(self):
        return self.sum / self.n

    def add(self,loss):
        self.sum += loss
        self.n += 1
        
def compute_accuracy(output,labels):
    predictions = torch.argmax(output, dim = 1)
    return torch.mean((predictions == labels).float())


def train_one_epoch(model,optimizer,dataloader):
    loss_aggregator = UpdatingMean()
    model.train()
    for x,y in tqdm(dataloader):
        
        optimizer.zero_grad()
        
        x = x.to(cuda0)
        y = y.to(cuda0)
        
        output = model(x)
        # compute_accuracy(output,y)
        loss = F.cross_entropy(output, y, reduction='mean')
        loss.backward()
        optimizer.step()
        loss_aggregator.add(loss.item())
        
        wandb.log({"Training loss": loss})
        
    
    return loss_aggregator.mean()

def run_validation_epoch(net,dataloader):
    
    accuracy_aggregator = UpdatingMean()
    loss_aggregator = UpdatingMean()
    # Put the network in evaluation mode.
    net.eval()
    # Loop over batches.
    for batch in tqdm(dataloader):
        # Forward pass only.
        output = net(batch[0].to(cuda0))

        # Compute the accuracy using compute_accuracy.
        accuracy = compute_accuracy(output, batch[1].to(cuda0))
        loss = F.cross_entropy(output, batch[1].to(cuda0),reduction='mean')
        #wandb.log({"Validating loss": loss,"Validating acc.": accuracy})

        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
        loss_aggregator.add(loss.item())

    return accuracy_aggregator.mean(),loss_aggregator.mean()


if __name__ == '__main__':
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    
    cuda0 = torch.device('cuda:0')
    #####hyperparameters###########
    BATCH_SIZE = 1024
    NUM_WORKERS = 0
    Learning_rate = 0.0001
    NUM_EPOCHS = 40
    Hidden_size = 128
    ################################
    
    # train_dataset = Dataset("../data",split='train')
    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size = BATCH_SIZE,
    #     num_workers = NUM_WORKERS,
    #     shuffle = True
    # )
    
    # eval_dataset = Dataset("../data",split='val')
    # eval_dataloader = DataLoader(
    #     eval_dataset,
    #     batch_size = BATCH_SIZE,
    #     num_workers = NUM_WORKERS,
    #     shuffle = False
    # )

    dataset = Dataset(path="../data/imgint_trainset_v2.hdf5")
    train_val_split = [int(0.7*dataset.num_pixels),dataset.num_pixels-int(0.7*dataset.num_pixels)]
    train_dataset, val_dataset = random_split(
        dataset = dataset,
        lengths=train_val_split
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = True
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )


    # model = RNN(4, Hidden_size, 2, 52).to(cuda0)
    model = LSTM(4, Hidden_size, 2, 52, True).to(cuda0)
    # model = GRU(4, Hidden_size, 2, 52, True).to(cuda0)
    
    
    optimizer = Adam(model.parameters(),lr = Learning_rate)
    wandb.init(project = 'lab3',name = 'Bi_'+model.codename+'_'+'hidden_size_'+str(Hidden_size))
    wandb.config.batch_size = BATCH_SIZE
    wandb.config.epochs = NUM_EPOCHS
    wandb.config.lr = Learning_rate
    wandb.hidden_size = Hidden_size
    
    
    best_accuarcy = 0
    for i in tqdm(range(NUM_EPOCHS)):
        loss = train_one_epoch(model, optimizer, train_dataloader)
        print('[Epoch %02d]Training Loss = %0.4f' %(i + 1, loss))
        
        #eval
        
        val_acc, val_loss = run_validation_epoch(model, eval_dataloader)
        print('[Epoch %02d]Validating Acc.: %.4f%%, Loss:%.4f' % (i + 1, val_acc * 100, val_loss))
        wandb.log({"epoch": i,"Validating loss": val_loss,"Validating acc.": val_acc})
        
        if val_acc > best_accuarcy:
            best_accuarcy = val_acc
            best_checkpoint = {
                'epoch_idx': i,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(best_checkpoint,f'checkpoints/bi/{model.codename}_'+'best'+'.pth')
        
        if i % 4 == 0:
            checkpoint = {
                'epoch_idx': i,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            dt=datetime.now()
            date=dt.strftime('%Y-%m-%d-%H-%M-%S')
            torch.save(checkpoint, f'checkpoints/bi/{model.codename}_'+'epoch_'+str(i+1)+str(date)+'.pth')
    
        
    print('Best validating acc. %.4f%%',best_accuarcy)
    wandb.log({
        "Best acc.":best_accuarcy
    })
    
        
        
        
    
    
    
    
    
    