import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import math
from tqdm import trange
import trimesh
import copy
import time
from tqdm import tqdm
from sklearn.neighbors import KDTree
import torch.nn.functional as F
import os
import itertools as it
knn_p=int(100)

def farthest_point_sample(xyz, pointnumber):
    device ='cuda'
    N, C = xyz.shape
    torch.seed()
    xyz=torch.from_numpy(xyz).float().to(device)
    centroids = torch.zeros(pointnumber, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)

    farthest[0]=N/2
    for i in tqdm(range(pointnumber)):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids.detach().cpu().numpy().astype(int)


def compute_stdavg(normallist,iz):

    length=len(normallist)
    normallist=np.abs(normallist)
    if(length<=2):
        return 1
    avg=np.mean(normallist,axis=0)
    avg=avg/np.linalg.norm(avg)

    coslist2=np.arccos(np.dot(normallist,avg))
    coslist2=np.nan_to_num(coslist2, nan=np.pi)

    return np.std(coslist2)
def rod_rotation(r,  theta):
    rx=r[0]
    ry=r[1]
    rz=r[2]
    M=np.array([
        [0,-rz,ry],
        [rz,0,-rx],
        [-ry,rx,0]
    ])

    R=np.eye(3) + np.sin(theta)*M + (1-np.cos(theta))*np.dot(M,M)
    return R

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

def prerotation(dd):
    itdn=np.zeros((4,3))
    itdn[0,0]=np.sin(dd / 180 * np.pi )
    itdn[1,0]=-np.sin(dd / 180 * np.pi )
    itdn[2,1]=np.sin(dd / 180 * np.pi )
    itdn[3,1]=-np.sin(dd / 180 * np.pi )
    itdn[0,2]=np.cos(dd / 180 * np.pi )
    itdn[1,2]=np.cos(dd / 180 * np.pi )
    itdn[2,2]=np.cos(dd / 180 * np.pi )
    itdn[3,2]=np.cos(dd / 180 * np.pi )
    rmat=np.zeros((4,3,3))
    rmat[0]=rotation_matrix_from_vectors([0,0,1],itdn[0])
    rmat[1]=rotation_matrix_from_vectors([0,0,1],itdn[1])
    rmat[2]=rotation_matrix_from_vectors([0,0,1],itdn[2])
    rmat[3]=rotation_matrix_from_vectors([0,0,1],itdn[3])
    return rmat
class Generator3D6(object):

    def __init__(self, model1, model2, device):
        self.model1 = model1
        self.model2 = model2
        self.device = device

    def upsample(self, xyzin,  para):

        pc = self.gp(xyzin,  para)

        return pc

    def gp(self, xyzin, para):

        xyzin=np.squeeze(xyzin,0)
        tree1=KDTree(xyzin)
        
        #----generate seedpoint-------------------------------------------#
        print("generate seedpoint")
        wq="./dense"
        wq=wq+" 0.004 "+str(xyzin.shape[0])+" 0.011 0.015 "
        print(wq)
        
        os.system(wq)
        data2=np.loadtxt("target.xyz")
        seedpoints=data2[:,0:3]

        if seedpoints.shape[0]< para['npoint']:
             print("no enough seed points")

        #-----direction estimation------------------------------------------#
        
     
        print("direction estimation")        
        patch_number=seedpoints.shape[0]//400
        p_split = np.array_split(seedpoints, patch_number, axis=0)
        normal=None
        for i in tqdm(range(len(p_split))):
            dist,  idx = tree1.query(p_split[i], knn_p)
            cloud=xyzin[idx]
            cloud=cloud-np.tile(np.expand_dims(p_split[i],1),(1,knn_p,1))

            with torch.no_grad():
                c = self.model1.encode_inputs(torch.from_numpy(np.expand_dims(cloud,0)).float().to(self.device))
            with torch.no_grad():
                n = self.model1.decode(c)
            n=n.detach().cpu().numpy()
            if normal is None:
                normal=n
            else:
                normal= np.append(normal,n,axis=0)

        #-----seeds rectification------------------------------------------#
        if para['SR'] is True:
            print("seeds rectification") 

            #select normals for each points and compute STD
            tripoint_idx=data2[:,3:6].astype(int)
            normalvector=[ [] for x in range(xyzin.shape[0])]
            stdvector=np.zeros(xyzin.shape[0])
            for i in range(tripoint_idx.shape[0]):
                for j in range(tripoint_idx.shape[1]):
                    normalvector[tripoint_idx[i][j]].append(list(normal[i]))
            for i in range(xyzin.shape[0]):
                stdvector[i]=compute_stdavg(normalvector[i],i)
         
            #compute the average STD for each seedpoints
            avgstd=stdvector[tripoint_idx]
            avgstd=(avgstd[:,0]+avgstd[:,1]+avgstd[:,2])/3.0

            #find three nearest seed points
            tripoint_idx=tripoint_idx.swapaxes(0,1)
            midpoint=xyzin[tripoint_idx]*0.5+np.tile(np.expand_dims(seedpoints,0),(3,1,1))*0.5

            seedtree=KDTree(seedpoints)
            SR_normal=np.zeros((3,normal.shape[0],normal.shape[1]))
            SR_differ=np.zeros((normal.shape[0],6))

            for k in range(3):
                _,idx=seedtree.query(midpoint[k])
                SR_normal[k]=np.squeeze(normal[idx])          

            #compute STD bewteen four normals 
            for i in range(normal.shape[0]):
                SR_differ[i][0]=np.arccos(np.dot(normal[i],SR_normal[0][i]))
                SR_differ[i][1]=np.arccos(np.dot(normal[i],SR_normal[1][i]))                    
                SR_differ[i][2]=np.arccos(np.dot(normal[i],SR_normal[2][i]))
                SR_differ[i][3]=np.arccos(np.dot(SR_normal[0][i],SR_normal[1][i]))  
                SR_differ[i][4]=np.arccos(np.dot(SR_normal[0][i],SR_normal[2][i]))
                SR_differ[i][5]=np.arccos(np.dot(SR_normal[1][i],SR_normal[2][i]))
            SR_differ=np.nan_to_num(SR_differ, nan=np.pi)
            SR_std = np.std(SR_differ,1)       

            # compare and preserve
            preserve_index=np.where(SR_std<(avgstd*2))[0] 

        #-----distance estimation------------------------------------------#
        print("distance estimation")         
        
        n_split = np.array_split(normal, patch_number, axis=0)
        xyzout=[]
        length=None

        for i in tqdm(range(len(p_split))):
            
            dist,  idx = tree1.query(p_split[i], knn_p)
            cloud=xyzin[idx]
            cloud=cloud-np.tile(np.expand_dims(p_split[i],1),(1,knn_p,1))

            for j in range(cloud.shape[0]):
                M1=rotation_matrix_from_vectors(n_split[i][j],[1,0,0])
                cloud[j]=(np.matmul(M1,cloud[j].T)).T

            with torch.no_grad():
                c = self.model2.encode_inputs(torch.from_numpy(np.expand_dims(cloud,0)).float().to(self.device))
            with torch.no_grad():
                n = self.model2.decode(c)
            L=np.tile(np.expand_dims(n.detach().cpu().numpy(),1),(1,3))
            if length is None:
                length=L
            else:
                length= np.append(length,L,axis=0)
        
        xyzout=seedpoints+normal*length

        #------apply seeds rectification-----------------------------------------#
        if para['SR'] is True:
            xyzout=xyzout[preserve_index]
            normal=normal[preserve_index]
            length=length[preserve_index]
            seedpoints=seedpoints[preserve_index]
            avgstd=avgstd[preserve_index]

        #-------remove outliers----------------------------------------#
        if para['remove'] is True:
            print("remove outliers")  
            tree3=KDTree(xyzout)
            dist, idx = tree3.query(xyzout, 30)
            avg=np.mean(dist,axis=1)
            avgtotal=np.mean(dist)
            idx=np.where(avg<avgtotal*1.5)[0]
            xyzout=xyzout[idx]
            normal=normal[idx]
            length=length[idx]
            seedpoints=seedpoints[idx]
            avgstd=avgstd[idx]

        #-------farthest point sample----------------------------------------#        
        print("farthest point sample")              
        centroids=farthest_point_sample(xyzout, para['npoint'])
        xyzout=xyzout[centroids]
        normal=normal[centroids]
        length=length[centroids]
        seedpoints=seedpoints[centroids]   
        avgstd=avgstd[centroids]

        #-------distance rectification------------------------------------#

        if para['DR'] is True:
            print("distance rectification") 
            #For each seed point, generate a plane across the point and perpendicular to its direction
            plane_para=np.zeros((normal.shape[0],normal.shape[1]+1))
            plane_para[:,0:3]=normal.copy()
            plane_para[:,3]=np.sum(-normal*seedpoints,axis=1)

            #compute the intersecton of the plane and line (x=1, y=1, z=t)
            CC_normal=np.zeros((normal.shape[0],normal.shape[1],para['DR_number']))
            CC_normal[:,0,0]=1
            CC_normal[:,1,0]=1
            CC_normal[:,2,0]=-(plane_para[:,0]+plane_para[:,1]+plane_para[:,3])/plane_para[:,2]
            #use the vector from the seedpoint to the intersecton as the direction of cc_1
            CC_normal[:,:,0]=CC_normal[:,:,0]-seedpoints
            
            #if the plane is parallel to (x=1, y=1, z=t), instead the direction with [0,0,1]
            for i in range(CC_normal.shape[0]):
                if plane_para[i][2]==0 :
                    CC_normal[i,0,0]=0
                    CC_normal[i,1,0]=0
                    CC_normal[i,2,0]=1

            #normalization
            templength=np.linalg.norm(CC_normal[:,:,0],axis=1)
            templength=np.tile(np.expand_dims(templength,1),(1,3))
            CC_normal[:,:,0]=CC_normal[:,:,0]/templength


            #get the direction of each cc_i by rotation            
            DR_angle=360/para['DR_number']
            for i in range(CC_normal.shape[0]):
                rmat=rod_rotation(normal[i],DR_angle/180*np.pi)
                for j in range(1,para['DR_number']):
                    CC_normal[i,:,j]=(rmat@CC_normal[i,:,j-1].T).T


            Alist=np.maximum(0.5-avgstd,0.3)*100
            Alist=np.tile(np.expand_dims(Alist,1),(1,3))

            #computd the length of cc_i
            CC_length=np.tile(np.expand_dims(length*np.tan(Alist/180*np.pi),2),(1,1,para['DR_number']))

            #generate new seed points
            DR_seedpoints=np.tile(np.expand_dims(seedpoints,2),(1,1,para['DR_number'])) 
            DR_seedpoints=DR_seedpoints+(CC_normal*CC_length)


            #generate new direction
            DR_xyzout=np.tile(np.expand_dims(xyzout,2),(1,1,para['DR_number']))
            DR_normal=(DR_xyzout-DR_seedpoints)
            templength=np.linalg.norm(DR_normal,axis=1)
            templength=np.tile(np.expand_dims(templength,1),(1,3,1))
            DR_normal=DR_normal/templength

            patch_number=para['npoint']/400

            for k in range(para['DR_number']):
                DR_p_split = np.array_split(DR_seedpoints[:,:,k].copy(), patch_number, axis=0)
                DR_n_split = np.array_split(DR_normal[:,:,k].copy(), patch_number, axis=0)
                length=None
                for i in tqdm(range(len(DR_p_split))):

                    dist,  idx = tree1.query(DR_p_split[i], knn_p)
                    cloud=xyzin[idx]
                    cloud=cloud-np.tile(np.expand_dims(DR_p_split[i],1),(1,knn_p,1))

                    for j in range(cloud.shape[0]):
                        M1=rotation_matrix_from_vectors(DR_n_split[i][j],[1,0,0])
                        cloud[j]=(np.matmul(M1,cloud[j].T)).T

                    with torch.no_grad():
                        c = self.model2.encode_inputs(torch.from_numpy(np.expand_dims(cloud,0)).float().to(self.device))
                    with torch.no_grad():
                        n = self.model2.decode(c)
                    L=np.tile(np.expand_dims(n.detach().cpu().numpy(),1),(1,3))
                    if length is None:
                        length=L
                    else:
                        length= np.append(length,L,axis=0)
                tempcloud=DR_seedpoints[:,:,k]+length*DR_normal[:,:,k]

                xyzout=xyzout+tempcloud
            xyzout=xyzout/(para['DR_number']+1)
     
        return xyzout
