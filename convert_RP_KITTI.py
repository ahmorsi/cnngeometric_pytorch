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

use_cuda = torch.cuda.is_available()
# Argument parsing
parser = argparse.ArgumentParser(description='CNN Relative Pose Estimation to KITTI Format')
# Paths
parser.add_argument('--model', type=str, default='trained_models/best_checkpoint_resnet18_adam_pose_mse_loss.pth.tar', help='Trained affine model filename')
#parser.add_argument('--model-tps', type=str, default='trained_models/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar', help='Trained TPS model filename')
parser.add_argument('--path', type=str, default='/media/develop/Ahmed-HD/Thesis/KITTI/dataset/sequences/', help='Path to PF dataset')
parser.add_argument('--pairs', type=str, default='/media/develop/Ahmed-HD/Thesis/KITTI/dataset/sequences/00_pairs.csv', help='Path to PF dataset')
args = parser.parse_args()

dataset_path=args.path
dataset_pairs_file = args.pairs
# Create model
print('Creating CNN model...')

model = CNNGeometric(use_cuda=use_cuda,geometric_model='pose',arch = 'resnet18')

print('Load CNN Weights ...')
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])

# Dataset and dataloader
dataset = PairsDataset(csv_file=dataset_pairs_file,
                    training_image_path=dataset_path,
                    output_size=(240, 240),
                    transform=NormalizeImageDict(['source_image','target_image']))

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=False, num_workers=1)

batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)
accum_pose = np.identity(4)
with open("estimated_pose2.txt",'w') as f:
    for img_src_name,img_target_name,batch in dataloader:

        batch = batchTensorToVars(batch)
        model.eval()
        theta_pose,_,_ = model(batch)

        data_np = theta_pose.data.numpy()
        R = data_np[0,0:9].reshape(3,3)
        T = data_np[0,9:].reshape(3,1)
        Tmat = np.hstack((R,T))
        Tmat_h = np.vstack((Tmat,np.array([0,0,0,1])))
        accum_pose = np.dot(accum_pose,Tmat_h)
        Tmat_1d = accum_pose[:-1].reshape((12,))
        f.write(" ".join([str(x) for x in Tmat_1d]))
        f.write("\n")
        print(img_src_name,img_target_name)

