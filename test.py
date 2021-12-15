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
from train import UpdatingMean,compute_accuracy
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, classification_report


def eval_on_test(net,dataloader,class_ids):
    accuracy_aggregator = UpdatingMean()
    net.eval()
    true_label_list = []
    pred_label_list = []
    target_names = ['Maize', 'Meadow', 'Pasture', 'Potatoes', 'Spelt', 'Sugar beet', 'Sunflowers', 'Vegetables', 'Vines', 'Wheat', 'Winter barley', 'Winter rapeseed', 'Winter wheat']
    for batch in tqdm(dataloader):
        
        x = batch[0].to(cuda0).squeeze(0)
        label = batch[1].to(cuda0).squeeze(0)
        output = net(x)
        output = torch.nn.functional.log_softmax(output)
        #convert ids 0-12 to original index in the dataset 0-55
        
        output = torch.argmax(output, dim = 1)
        pred = torch.take(class_ids, output)
        
        # filter out class 0
        mask = label!=0
        pred = pred[mask]
        label = label[mask]
        true_label_list.append(label.cpu().detach().numpy())
        pred_label_list.append(pred.cpu().detach().numpy())

        # Compute the accuracy using compute_accuracy.
        accuracy = torch.mean((pred == label).float())
        
       
        # Save accuracy value in the aggregator.
        accuracy_aggregator.add(accuracy.item())
    
    weight = [126545,  363550 ,  81047  ,  8580 ,  10343 ,  37218 ,  11157  , 47226, 6568 ,  16840 ,  46867  , 43239 , 127351]
    weight = np.asarray(weight)
    weight = weight/np.sum(weight)
    y_true = np.concatenate(true_label_list)
    y_pred = np.concatenate(pred_label_list)
    accuracy = accuracy_score(y_true,y_pred)
    precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='micro')[:-1]
    Confusion_M = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred,target_names=target_names)
    print("confusion matrix\n", Confusion_M)
    print("accuracy: ", accuracy)
    print("precision: ",precision)
    print("recall: ",recall)
    print("f1: ",f1)
    print(report)
    return accuracy_aggregator.mean()


if __name__ == '__main__':
    random_seed = 1
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    
    cuda0 = torch.device('cuda:0')
    #####hyperparameters###########
    BATCH_SIZE = 1
    NUM_WORKERS = 12
    Hidden_size = 128
    dropout = 0.3
    ################################
    
    model = LSTM(4, Hidden_size, 2, 13, True, dropout=dropout)
    model.load_state_dict(torch.load('./checkpoints/dropout/LSTM_epoch_9_dropout_0.3.pth')['net'])
    model.to(cuda0)
    
    testset =  Dataset(path="../data/imgint_testset_v2.hdf5", split='test')
    test_dataloader =  DataLoader(
        testset,
        batch_size = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        shuffle = False
    )
    class_ids = [20, 21, 27, 30, 36, 38, 42, 45, 46, 48, 49, 50, 51]
    class_ids = torch.from_numpy(np.array(class_ids)).to(cuda0)
    
    eval_on_test(model, test_dataloader, class_ids)
    