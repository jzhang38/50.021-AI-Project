# Video Resotoration on Historical Footage 

A Course Project for SUTD 50.021 Artificial Intelligence (2022)

Created by [Mark He Huang](https://markhh.com/), [Peiyuan Zhang](https://www.linkedin.com/in/lance-peiyuan-zhang-5b2886194/), [Pheh Jing Jie](https://www.linkedin.com/in/phehjingjie/?originalSubdomain=sg), [Yucheng Ma](https://www.linkedin.com/in/kevin-ma-yuchen/?originalSubdomain=sg).


## Setup Environment

```bash
# Set up Python virtual environment
conda create --name envname python=3.8 -y
conda activate envname
conda install pytorch=1.10 torchvision cudatoolkit=11.3 -c pytorch
pip3 install openmim
mim install mmcv-full==1.5.0
cd mmediting
pip3 install -e .
```

-   Please note that you must install mmediting using the code contained in this repo because we've modified some code from the mmediting official release.


## Setup Datasets

You may download the datasets used in the project from the following URLs:

-  [Super Resolution Dataset](https://drive.google.com/file/d/1tpcBmAuZ8zI178NB6l5f3YYmAUL2U9KZ/view)
-  [Video Denoising Dataset](https://drive.google.com/drive/folders/1cWUISEm_LnUMGb6MnttGzs8J1_8LEG1p)

After extracing the dataset, you will get custom_gt, custom_gt_small, custom_low_res. Please put them under a common directory, and specify the path to that directory in shard_dataset.py,
mmediting/configs/restorers/basicvsr/basicvsr_reds4_custom_denoise.py, mmediting/configs/restorers/basicvsr/basicvsr_reds4_custom.py, mmediting/configs/restorers/basicvsr/basicvsr_reds4_custom_lr_small.py, mmediting/configs/restorers/basicvsr/basicvsr_reds4_custom_load.py, accordingly. 

custom_gt and custom_low_res are used for video super resolution, where custom_gt is four times the size of custom_low_res.

custom_gt_small and custom_low_res are used for video super resolution, where custom_gt_small is of the same size as custom_low_res.


The next step is to shard the dataset into 100 frames subfolder using
```
python shard_dataset.py
```

## Model Checkpoints

You may download the model checkpoints from here: [Google Drive](https://drive.google.com/drive/folders/1mSN0ieYxZxASq7ONMqvtrqxkxjSn_iRt?usp=sharing). 

## Training
To download [BasicVSR](https://arxiv.org/abs/2012.02181)'s checkpoint trained on the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset
```
wget https://download.openmmlab.com/mmediting/restorers/basicvsr/basicvsr_reds4_20120409-0e599677.pth
mv basicvsr_reds4_20120409-0e599677.pth basicvsr_reds4_no_optimizer
```
To finetune [BasicVSR](https://arxiv.org/abs/2012.02181) model on our historical footage dataset based on the REDS checkpoint
```
CUDA_VISIBLE_DEVICES=3 ./tools/dist_train_load.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_load.py 1 
```
To train [BasicVSR](https://arxiv.org/abs/2012.02181) model from scratch on our  historical footage dataset,
```
CUDA_VISIBLE_DEVICES=1 ./tools/dist_train.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom.py 1
```
To finetune our modified [BasicVSR](https://arxiv.org/abs/2012.02181) model for denoising on our  historical footage dataset based on [BasicVSR REDS checkpoint](https://github.com/open-mmlab/mmediting/blob/master/configs/restorers/basicvsr/basicvsr_reds4.py),
```
CUDA_VISIBLE_DEVICES=0 ./tools/dist_train_load.sh ./configs/restorers/basicvsr/basicvsr_reds4_custom_denoise.py 1 
```
Note that some parameter cannot be loaded into the model because we have changed some module's dimension to keep the output image size the same as the input size (for denoising).
## Evaluation

By default, we directly use test set for training validation, and the checkpoins/ evaluation resuls will be automatically saved in the [`mmediting/work_dirs`](mmediting/work_dirs) directory during training. 

To evaluate on the test set using trained model, you can run the following command:

```bash
CUDA_VISIBLE_DEVICES=1 ./tools/dist_test.sh [config path] [checkpoint path] 1
```


