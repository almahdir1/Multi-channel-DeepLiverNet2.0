# Multi-channel-DeepLiverNet2.0
Liver project
by Redha Ali

This is the official implementation of our proposed DeepLiverNet2.0:
![image](https://github.com/almahdir1/Multi-channel-DeepLiverNet2.0/blob/main/Figures/DeepLiverNet2.png)

Tensorflow implementation of Swin Transformer model.
Based on [Official Tensorflow implementation](https://github.com/almahdir1/Swin-Transformer-TF).
![image](https://user-images.githubusercontent.com/24825165/121768619-038e6d80-cb9a-11eb-8cb7-daa827e7772b.png)

## Requirements
- `tensorflow >= 2.6.0`
- `scikit-image==0.17.2`
- `scipy==1.7.1`
- `numpy`
- `matplotlib==3.5.1`
- `os==2.1.4`
- `MATLAB 2021a or later`
or use `environment.yml`

## Pretrained Swin Transformer Checkpoints
**ImageNet-1K and ImageNet-22K Pretrained Checkpoints**  
| name | pretrain | resolution |acc@1 | #params | model |
| :---: | :---: | :---: | :---: | :---: | :---: |
|`swin_base_224` |ImageNet-22K|224x224|85.2|88M|[github](https://github.com/rishigami/Swin-Transformer-TF/releases/download/v0.1-tf-swin-weights/swin_base_224.tgz)|

## Data Preparation
To extract the 11 sileces please run the script "Create_11_Slices_dataset.m" file 

## Deep Features Extraction
To extract the features from the 11 sliecse using pre-trained Swin transfomer please run the script "run_Swin_Fea_Ext.py"

## Ten Fold Cross Validation Classification
TensorFlow: To train and test the Multi-channel-DeepLiverNet2.0 please use the script "train.py" and "test.py"
MATLAB: To train the Multi-channel-DeepLiverNet2.0 please run the script "run_train_10k_Fold_CV_classification.m"


# License
Multi-channel-DeepLiverNet2.0 is released under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International Public License (CC BY-NC-ND 4.0).

Copyright © 2024 Cincinnati Children's Hospital Medical Center

