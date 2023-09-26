# PointNet and PointNet++ for TUM-Facade

This repo is PointNet++ and PointNet semantic segmentation implementation for TUM-Facade. It is based off the repo by Xu Yan PointNet++ pytorch https://github.com/yanx27/Pointnet_Pointnet2_pytorch. It is usable in Google Colab 2023.


## Install following modules and packages
pip install laspy

pip install scikit-learn

pip install matplotlib

pip install pytz

pip install torchviz

pip install open3d


## Dataset of TUM-Facade
Download from https://github.com/OloOcki/tum-facade.

## Run in Google Colab

Example 1:

%run sem_seg_training.py  --save True --epoch 25 --rootdir 'YOUR FILE DIRECTORY' --test_area 'DEBY_LOD2_4959323.las'

!python sem_seg_testing.py  --save True ---rootdir 'YOUR FILE DIRECTORY' --visual --num_votes 1 --test_area 'DEBY_LOD2_4959323.las'

Example 2:

%run sem_seg_training.py  --save True --epoch 25 --RGB_OFF --rootdir 'YOUR FILE DIRECTORY' --test_area 'DEBY_LOD2_4959323.las'

!python sem_seg_testing.py  --save True --RGB_OFF --rootdir 'YOUR FILE DIRECTORY' --visual --num_votes 1 --test_area 'DEBY_LOD2_4959323.las'


## Reference
[yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)




