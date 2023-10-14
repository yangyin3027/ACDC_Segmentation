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
```python
torchrun --standalone --nnodes=1 --nproc_per_node=3 ./src/main.py --data [data_root_dir] --batch 24 --lr 0.01 --init kaiming --loss dicece
```
# Inference
```python
python ./src/inference.py --model attenunet --checkpoint [dir/checkpoint.pth.tar] --type polot --data [data_root_dir]
```
# Examples
![Alt text](https://github.com/yangyin3027/ACDC_Segmentation/blob/main/src/examples/predicted.png)

# License
MIT License


