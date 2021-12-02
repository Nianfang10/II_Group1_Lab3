# II_Group1_Lab3

Code for Image Interpretation Lab 3 at ETH Zurich FS21.  
Team member: **Liyuan Zhu, Rushan Wang, Nianfang Shi**  

### Goals
Using recurrent neural network (RNN) to automatically label 13 crop types from Sentinel-2 image data and generate map based on predictions.

### Dataset
We divided the original trainset _(imgint_trainset_v2.hdf5)_ into 2 parts. We used 70% pixels as the training dataset, and the other 30% as our validation dataset.

### Models
We tried three models: **RNN**, **GRU**, and **LSTM**

___

### Quantitative Results
**Best results achieved on each model:**
| Models | RNN|GRU|LSTM
| ------ | ----- | ----| -----|
| Accuracy |- | - | - |

**Results with different parameters on RNN:**
| Learning rate (t.b.d) | 0.0001|0.0002|0.002
| ------ | ----- | ----| -----|
| Accuracy      |- | - | - |

Training time:  
Inference time:  
Memory requirements:  

**Results with different parameters on GRU:**
| Batch size (t.b.d)| 128|512|1028
| ------ | ----- | ----| -----|
| Accuracy      |- | - | - |

Training time:  
Inference time:  
Memory requirements:  

**Results with different parameters on GRU:**
| Batch size (t.b.d)| 128|512|1028
| ------ | ----- | ----| -----|
| Accuracy      |- | - | - |

Training time:  
Inference time:  
Memory requirements:  

### Qualitative Results
Prediction map:
