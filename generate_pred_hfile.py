import numpy as np
import h5py
from PIL import Image
import torch
import pdb
from networks import LSTM
from dataset import Dataset
from torch.utils.data import DataLoader
# Open hdf5 file
# testset = "../data/imgint_testset_v2.hdf5"
# testset = h5py.File(testset, "r")

# # # Read data shape and labels
# # test_set_shape = testset["data"].shape #(8358, 24, 24, 71, 4)
# testset_target = testset["gt"][...]



pred_path = "../data/pred.hdf5"
predset = h5py.File(pred_path,'a')
# if 'gt' not in predset.keys():
#     predset.create_group('gt')
del predset["gt"]
predset['gt'] = np.zeros(predset['data'].shape[:3])
# pdb.set_trace()
# pred['data'] = np.zeros(test_set_shape)
# pred.create_group('data')
# testset.copy('data',pred)



#####hyperparameters###########
BATCH_SIZE = 1
NUM_WORKERS = 12
Hidden_size = 128
################################
cuda0 = torch.device('cuda:0')

model = LSTM(4, Hidden_size, 2, 13, True)
model.load_state_dict(torch.load('./checkpoints/dropout/LSTM_best_dropout_0.3.pth')['net'])
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

i = 0
for x,label in test_dataloader:
    x = x.to(cuda0).squeeze(0)
    label = label.to(cuda0).squeeze(0)
    output = model(x)
    output = torch.nn.functional.softmax(output)
    #convert ids 0-12 to original index in the dataset 0-55
    
    output = torch.argmax(output, dim = 1)
    mask = label==0
    pred = torch.take(class_ids, output)
    pred[mask] = 0
    
    
    pred  = pred.view(24,24).cpu().detach().numpy()
    
    predset['gt'][i] = pred
    i += 1
    
    

predset.close()