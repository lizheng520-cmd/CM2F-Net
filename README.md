# CM2F-Net
This is the codebase for our paper "Cross-Modal Multi-Level Feature Fusion Method for Continuous Sign Language Recognition" which is currently under review at *The Visual Computer*. 
All datasets analysed during this study are available in links. 

## Prerequisites

- This project is implemented in Pytorch (better >=1.13 to be compatible with ctcdecode or these may exist errors). Thus please install Pytorch first.

- ctcdecode==0.4 ，for beam search decode.


## Data Preparation
PHOENIX-2014 dataset is openly available at [download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
After finishing dataset download, extract it to ./dataset/phoenix, it is suggested to make a soft link toward downloaded dataset
```
ln -s PATH_TO_DATASET/phoenix2014-release ./dataset/phoenix2014
```
Run the following command to generate gloss dict.
```
cd ./preprocess
python data_preprocess.py --process-image --multiprocessing
```
PHOENIX-2014-T dataset is openly available at [download link](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/).
After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
```
ln -s PATH_TO_DATASET/PHOENIX-2014-T-release-v3/PHOENIX-2014-T ./dataset/phoenix2014-T
```
Run the following command to generate gloss dict.
```
cd ./preprocess
python data_preprocess-T.py --process-image --multiprocessing
```
CSL-Daily dataset is openly available at [download link](https://ustc-slr.github.io/datasets/2021_csl_daily/).
After finishing dataset download, extract it. It is suggested to make a soft link toward downloaded dataset.
```
ln -s PATH_TO_DATASET ./dataset/CSL-Daily
```
Run the following command to generate gloss dict.
```
cd ./preprocess
python data_preprocess-CSL-Daily.py --process-image --multiprocessing
```

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --config ./configs/baseline.yaml --device your_device`
