
# PU-SSAS


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
coming soon

## Dataset
coming soon

## Evaluation
The code for evaluation can be download from:
```
https://github.com/pleaseconnectwifi/Meta-PU/tree/master/evaluation_code
https://github.com/jialancong/3D_Processing
```

## Acknowledgement
The code is based on [occupancy_networks](https://github.com/autonomousvision/occupancy_networks/) and [DGCNN](https://github.com/WangYueFt/dgcnn), If you use any of this code, please make sure to cite these works.
