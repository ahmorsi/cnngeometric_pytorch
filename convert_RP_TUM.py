from model.cnn_geometric_model import CNNGeometric
import numpy as np
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.pairs_dataset import PairsDataset
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars, str_to_bool
from geotnf.transformation import GeometricTnf
import math
import cv2

fx = 517.3
fy = 516.6
cx = 318.6
cy = 255.3

def convertMatrixToQuaternion(Mat4x4):
    qw = math.sqrt(max(0,1 + Mat4x4[0,0] + Mat4x4[1,1] + Mat4x4[2,2]))/2.0
    qx = math.sqrt(max(0,1 + Mat4x4[0,0] - Mat4x4[1,1] - Mat4x4[2,2]))/2.0
    qy = math.sqrt(max(0, 1 - Mat4x4[0, 0] - Mat4x4[1, 1] - Mat4x4[2, 2])) / 2.0
    qz = math.sqrt(max(0, 1 - Mat4x4[0, 0] - Mat4x4[1, 1] + Mat4x4[2, 2])) / 2.0

    return qx,qy,qz,qw


# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R
use_cuda = torch.cuda.is_available()
# Argument parsing
parser = argparse.ArgumentParser(description='CNN Relative Pose Estimation to TUM Format')
# Paths
parser.add_argument('--model', type=str, default='trained_models/best_checkpoint_adam_6d_pose_mse_loss.pth.tar', help='Trained affine model filename')
#parser.add_argument('--model-tps', type=str, default='trained_models/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar', help='Trained TPS model filename')
parser.add_argument('--path', type=str, default='/media/develop/Elements/aba4hi/Dataset/RGBD_Tracking/rgbd_dataset_freiburg1_360', help='Path to PF dataset')
parser.add_argument('--pairs', type=str, default='/media/develop/Elements/aba4hi/Dataset/RGBD_Tracking/rgbd_dataset_freiburg1_360/seq_pairs.csv', help='Path to PF dataset')
parser.add_argument('--out',type=str)
parser.add_argument('--abs',action='store_true')
args = parser.parse_args()

dataset_path=args.path
dataset_pairs_file = args.pairs
output_filename = args.out
# Create model
print('Creating CNN model...')

model = CNNGeometric(use_cuda=use_cuda,geometric_model='pose',arch = 'resnet18')

is_absoulte = args.abs

print('Load CNN Weights ...')
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

# Dataset and dataloader
dataset = PairsDataset(csv_file=dataset_pairs_file,
                    training_image_path=dataset_path,
                    output_size=(240, 240),
                    use_timestamp=True,
                    transform=NormalizeImageDict(['source_image','target_image']))

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=1)

batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)
accum_pose = np.identity(4)

K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
K_t = K.transpose()
output_file = os.path.join(dataset_path,output_filename)
with open(output_file,'w') as f:
    for img_src_name,img_target_name,batch in dataloader:

        timestamp = batch['timestamp'].numpy()[0]
        batch = batchTensorToVars(batch)
        model.eval()
        theta_pose,_,_ = model(batch)

        data_np = theta_pose.data.numpy()
        R_theta = data_np[0,:3]
        T = data_np[0,3:].reshape((3,1))
        R = eulerAnglesToRotationMatrix(R_theta)
        # E = np.array([
        #     [0,data_np[0,0],data_np[0,1]],
        #     [data_np[0,2],0,data_np[0,3]],
        #     [data_np[0,4],data_np[0,5],0]
        # ])
        #F = K_t*E*K
        #R1,R2,t = cv2.decomposeEssentialMat(F)#cv2.decomposeEssentialMat(E)
        #R = data_np[0,0:9].reshape(3,3)
        #T = data_np[0,9:].reshape(3,1)
        #R = R1
        #T = t
        Tmat = np.hstack((R,T))
        Tmat_h = np.vstack((Tmat,np.array([0,0,0,1])))
        if is_absoulte:
            accum_pose = np.dot(accum_pose, Tmat_h)
        else:
            accum_pose = Tmat_h

        qx, qy, qz, qw = convertMatrixToQuaternion(accum_pose)
        T = accum_pose[:-1, 3]
        f.write("{0} {1} {2} {3} {4} {5} {6} {7}\n".format(
            timestamp,
            T[0],T[1],T[2],
            qx, qy, qz, qw
        ))
        #Tmat_1d = accum_pose[:-1].reshape((12,))
        #f.write(" ".join([str(x) for x in T]))
        #f.write(" {0} {1} {2} {3}".format(qx,qy,qz,qw))
        #f.write("\n")
        print("{0} {1} {2}".format(timestamp,img_src_name[0],img_target_name[0]))
        #print(img_src_name,img_target_name)

