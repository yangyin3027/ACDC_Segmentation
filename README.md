# ACDC_Segmentation
Segmentation Models for Automated Cardiac Diagnosis Challenge adopted with Attention Unet Architecture. Part of the code were inspired from [HiLab-git/SSL4MIS](https://github.com/HiLab-git/SSL4MIS.git), and [LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation.git). The ACDC datasets and compute metrics code were downloaded and adapted from the [Automated Cardiac Diagnosis Challenge (ACDC)](https://www.creatis.insa-lyon.fr/Challenge/acdc/)
# Model Architecture
![Alt text](https://github.com/yangyin3027/ACDC_Segmentation/blob/main/src/examples/attenunet.png)
Please see the original paper [here](https://arxiv.org/abs/1804.03999) for more details. 
# Environment Setup
There are two approaches:
  1. Using pip
  ```python
  pip install -r requirements.txt
  ```
  2. Using conda
  ```
  conda env create -n [env_name] --file environments.yml
  ```
# Train the model
Tricks to improve training performance:
- Use combination of CrossentropyLoss and DiceLoss
- Loss calculated directly on logits output of the model 
- LabelSmoothing with 0.1 significantly improve the significance
- Adam optimizer combined with linear Warmup significantly accelerate the model converge, 10x faster than SGD with similar performance
```python
torchrun --standalone --nnodes=1 --nproc_per_node=3 ./src/main.py --data [data_root_dir] --batch 24 --lr 0.01 --init kaiming --loss dicece
```
# Inference
```python
python ./src/inference.py --model attenunet --checkpoint [dir/checkpoint.pth.tar] --type polot --data [data_root_dir]
```
# Performance
End Diastolic (ED)
|Model | Mean Dice LV| Mean Dice RV| Mean Dice Myocardium| 
| :----|:-----------:|:------------:|:------------------:|
|AttenUnet|  0.967 | 0.938 | 0.889|

End Systolic (ES)
|Model | Mean Dice LV| Mean Dice RV| Mean Dice Myocardium| 
| :----|:-----------:|:------------:|:------------------:|
|AttenUnet|  0.914 | 0.883 | 0.899|

# Examples
![Alt text](https://github.com/yangyin3027/ACDC_Segmentation/blob/main/src/examples/predicted.png)

# License
MIT License


