
import sys
# import torch.distributions as dist
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import shutil
import argparse
from tqdm import tqdm
import time
import trimesh
import random
from collections import defaultdict
import fd.config
import fn.config
import fn.checkpoints
import fd.checkpoints
from generation import Generator3D6
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(20)
datadir='test/'
outdir='testout/'


datalist=[
'cow.xyz',
'coverrear_Lp.xyz',
'chair.xyz',
'camel.xyz',
'casting.xyz',
'duck.xyz',
'eight.xyz',
'elephant.xyz',
'elk.xyz',
'fandisk.xyz',
'genus3.xyz',
'horse.xyz',
'Icosahedron.xyz',
'kitten.xyz',
'moai.xyz',
'Octahedron.xyz',
'pig.xyz',
'quadric.xyz',
'sculpt.xyz',
'star.xyz']



scalelist=np.array([4])


para={
    'SR' :True,
    'DR' :True,
    'remove' :True,
    'npoint': 0,
    'DR_angle' :35,
    'DR_number' :5
}

if(len(sys.argv)==3):
    para['DR_angle']=int(sys.argv[1])
    para['DR_number']=int(sys.argv[2])
print(para['DR_angle'],para['DR_number'])

is_cuda = (torch.cuda.is_available())
device = torch.device("cuda" if is_cuda else "cpu")

out_dir= 'out/pointcloud/opu'

cfg1 = fn.config.load_config('config/fn.yaml')
cfg2 = fd.config.load_config('config/fd.yaml')

model = fn.config.get_model(cfg1, device)
model2 = fd.config.get_model(cfg2, device)

checkpoint_io1 = fn.checkpoints.CheckpointIO('out/fn', model=model)
load_dict =checkpoint_io1.load( 'model_best.pt')

checkpoint_io2 = fd.checkpoints.CheckpointIO('out/fd', model=model2)
load_dict =checkpoint_io2.load( 'model_best.pt')

model.eval()
model2.eval()

generator=Generator3D6(model, model2, device)

for w in range(scalelist.shape[0]):
    scalepath=outdir+str(scalelist[w])+"x_"+str( para['DR_angle'])+"_" +str( para['DR_number'])+"/"
    print(scalepath)
    if not os.path.exists(scalepath):
        os.makedirs(scalepath)
    for k in range(len(datalist)):
        print("processing "+datalist[k]+" scale "+str(scalelist[w]))
        #normalization
        xyzname=datadir+datalist[k]
        cloud =np.loadtxt(xyzname)
        cloud=cloud[:,0:3]
        bbox=np.zeros((2,3))
        bbox[0][0]=np.min(cloud[:,0])
        bbox[0][1]=np.min(cloud[:,1])
        bbox[0][2]=np.min(cloud[:,2])
        bbox[1][0]=np.max(cloud[:,0])
        bbox[1][1]=np.max(cloud[:,1])
        bbox[1][2]=np.max(cloud[:,2])
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max()
        scale1 = 1/scale
        for i in range(cloud.shape[0]):
            cloud[i]=cloud[i]-loc
            cloud[i]=cloud[i]*scale1
            
        para['npoint']=int(scalelist[w]*cloud.shape[0])
        print(para['npoint'])
        np.savetxt("test.xyz",cloud)


        
        cloud=np.expand_dims(cloud,0)

        pointcloud = np.array(generator.upsample(cloud, para))
        for i in range(pointcloud.shape[0]):
            pointcloud[i]=pointcloud[i]*scale
            pointcloud[i]=pointcloud[i]+loc

        np.savetxt(scalepath+datalist[k],pointcloud)
        
        print("done")

