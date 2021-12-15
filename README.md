# II_Group1_Lab3

Code for Image Interpretation Lab 3 at ETH Zurich FS21.  
Team member: **Liyuan Zhu, Rushan Wang, Nianfang Shi**  

## Goals
Using recurrent neural network (RNN) to automatically label 13 crop types from Sentinel-2 image data and generate map based on predictions.

## Dataset
We divided the original trainset _(imgint_trainset_v2.hdf5)_ into 2 parts. We used 70% pixels as the training dataset, and the other 30% as our validation dataset.

## Models
We tried three models: **RNN**, **GRU**, and **LSTM**


## Results
**Comparisons of three models**  
<img src="https://user-images.githubusercontent.com/91589561/146266610-f5fc7cc2-7947-49e6-b074-dd2d0c207e08.png" width="600"/>
  
  
### Parameter tuning on LSTM
**Bidirectional:**  
| LSTM | Onedirctional  |Bidirctional |
| ------ | -----|-----|
|Acc.(Validation) | 0.8684 |**0.8904** | 
  

**Learning rate:**
| Learning rate| 0.0001|0.001|0.005|0.01
| ------ | ----- | ----| -----|----|
| Acc.(Validation)      |0.8730 | **0.9461** | 0.9324 |0.8760
   

**Hidden size:**
| Hidden Size| 64|128|256
| ------ | ----- | ----| -----|
| Acc.(Validation)      |0.9207 | 0.9520 | **0.9652** |
  

**Dropout:**
| Dropout| 0|0.1|0.3|0.5
| ------ | ----- | ----| -----|----|
| Acc.(Validation)      |0.9461 |**0.9544** | 0.9512 |0.9478
  
  
  
### Prediction performance
<img src="https://user-images.githubusercontent.com/91589561/146269683-a794cc1b-23aa-41b3-bbe2-8c126e06d1f8.png" height="600"/>
  
  
  
### Map generation
**Prediction map:**  
<img src="https://user-images.githubusercontent.com/91589561/146266238-a9b05860-dcb0-4713-b2dc-b70552f462a6.png" height="800"/>

**Grount truth:**  
<img src="https://user-images.githubusercontent.com/91589561/146266273-af3a134a-c99e-4266-a4ff-12bd35d48204.png" height="800"/>

**Error map:**  
<img src="https://user-images.githubusercontent.com/91589561/146266363-eee4c143-8e24-4cd3-b0e9-5b777b4af3ab.png" height="800"/>


