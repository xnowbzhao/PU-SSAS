
# PU-SSAS
【Code of TPAMI paper】 

Self-Supervised Arbitrary-Scale Implicit Point Clouds Upsampling

[Paper address](https://ieeexplore.ieee.org/abstract/document/10159515)
 
## Environment
Pytorch 1.9.0

CUDA 10.2

## Evaluation
### a. Download models
Download the pretrained models from the link and unzip it to  `./out/`
```
https://drive.google.com/file/d/1W0t-Ea6ucJDQUt2mS_fUnvYiiMjZBRt4/view?usp=sharing
```
### b. Compilation
Run the following command for compiling dense.cpp which generates dense seed points
```
g++ -std=c++11 dense.cpp -O2 -o dense
```
### c. Evaluation
You can now test our code on the provided point clouds in the `test` folder. To this end, simply run
```
python generate.py
```
The 4X upsampling results will be created in the `testout` folder.

Ground truth are provided by [Meta-PU](https://drive.google.com/file/d/1dnSgI1UXBPucZepP8bPhfGYJEJ6kY6ig/view?usp=sharing)

## Training
Download the training dataset from the link and unzip it to /data/
```
https://pan.baidu.com/s/1yaacibc50d0dIWcW7OIxEA 
access code:lmwf

or

https://1drv.ms/f/s!AsP2NtMX-kUTmw44ZfSvhV_PLzxu?e=Y8iL97
```

Then run the following commands for training our network
```
python trainfn.py
python trainfd.py
```


## Generate Dataset

Download the pointclouds and watertight meshes from the link and unzip it to /data/

```
https://pan.baidu.com/s/1kWstsZMiZOJuGm5yvpNI3Q 
access code:208c

or

https://1drv.ms/f/s!AsP2NtMX-kUTmwxUMh-AZ5sJ7nl3?e=mHjhb9
```

Then run build.sh in /scripts/

If you want to generate the pointclouds and watertight meshes from other dataset, please follow the link: [occupancy_networks#building-the-dataset](https://github.com/autonomousvision/occupancy_networks#building-the-dataset)


## Evaluation Code
The code for evaluation can be download from:
```
https://github.com/pleaseconnectwifi/Meta-PU/tree/master/evaluation_code
https://github.com/jialancong/3D_Processing
```
## Citation
If the code is useful for your research, please consider citing:
  
    @ARTICLE{SelfPCU,
      author={Zhao, Wenbo and Liu, Xianming and Zhai, Deming and Jiang, Junjun and Ji, Xiangyang},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
      title={Self-Supervised Arbitrary-Scale Implicit Point Clouds Upsampling}, 
      year={2023},
      volume={},
      number={},
      pages={1-13},
      doi={10.1109/TPAMI.2023.3287628}}



## Acknowledgement
The code is based on [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/) and [DGCNN](https://github.com/WangYueFt/dgcnn), If you use any of this code, please make sure to cite these works.
