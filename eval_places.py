from __future__ import print_function, division
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.cnn_geometric_model import CNNGeometric
from data.pf_dataset import PFDataset
from data.places_dataset import PlacesDataset
from data.download_datasets import download_PF_willow
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars, str_to_bool,correct_keypoints
from util.cv_util import find_mutual_matached_keypoints,estimateAffineRansac,tensorPointstoPixels
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable
import cv2
import math

"""

Script to evaluate a trained model as presented in the CNNGeometric CVPR'17 paper
on the ProposalFlow dataset

"""

print('CNNGeometric PF evaluation script')

# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--model-aff', type=str, default='trained_models/best_streetview_checkpoint_adam_affine_grid_loss.pth.tar', help='Trained affine model filename')
parser.add_argument('--model-tps', type=str, default='trained_models/best_streetview_checkpoint_adam_tps_grid_loss.pth.tar', help='Trained TPS model filename')
parser.add_argument('--path', type=str, default='/home/develop/Work/Datasets/', help='Path to PF dataset')
parser.add_argument('--pairs', type=str, default='/home/develop/Work/Datasets/gardens_pairs_path_samples_sift_RANSAC_12kps.csv', help='Pairs CSV file')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

do_aff = not args.model_aff==''
do_tps = False#not args.model_tps==''
do_ransac = False
dataset_path=args.path
dataset_pairs_file = args.pairs
# Create model
print('Creating CNN model...')
if do_aff:
    model_aff = CNNGeometric(use_cuda=use_cuda,geometric_model='affine')
if do_tps:
    model_tps = CNNGeometric(use_cuda=use_cuda,geometric_model='tps')

# Load trained weights
print('Loading trained model weights...')
if do_aff:
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)
    model_aff.load_state_dict(checkpoint['state_dict'])
if do_tps:
    checkpoint = torch.load(args.model_tps, map_location=lambda storage, loc: storage)
    model_tps.load_state_dict(checkpoint['state_dict'])

# Dataset and dataloader
dataset = PlacesDataset(csv_file=dataset_pairs_file,
                    training_image_path=dataset_path,
                    transform=NormalizeImageDict(['source_image','target_image']))

if use_cuda:
    batch_size=16
else:
    batch_size=1

dataloader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=4)

batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

# Instantiate point transformer
pt = PointTnf(use_cuda=use_cuda)

# Instatiate image transformers
tpsTnf = GeometricTnf(geometric_model='tps',use_cuda=use_cuda)
affTnf = GeometricTnf(geometric_model='affine',use_cuda=use_cuda)

print('Computing PCK...')
total_correct_points_aff = 0
total_correct_points_tps = 0
total_correct_points_aff_tps = 0
total_points = 0
total_correct_points_ransac = 0

for i, batch in enumerate(dataloader):
   
    batch = batchTensorToVars(batch)
    
    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_im_shape_np = source_im_size.data.numpy()[0]
    target_im_shape_np = target_im_size.data.numpy()[0]

    source_points = batch['source_points']
    target_points = batch['target_points']
    
    # warp points with estimated transformations
    target_points_norm = PointsToUnitCoords(target_points,target_im_size)
    
    if do_aff:
        model_aff.eval()
    if do_tps:
        model_tps.eval()

    ransac_success = False
    # if do_ransac:
    #     source_points_np, target_points_np = source_points.data.numpy()[0].reshape((12,2)), target_points.data.numpy()[0].reshape((12,2))
    #     M = cv2.estimateRigidTransform(target_points_np,source_points_np,fullAffine=True)
    #     ransac_success = M is not None
    #     if ransac_success:
    #         #print("ransac_success")
    #         M_var = Variable(torch.FloatTensor(M.flatten()))
    #         # do affine only
    #         warped_points_ransac_norm = pt.affPointTnf(M_var,target_points)
    #         #warped_points_ransac_norm = pt.affPointTnf(M_var, target_points_norm)
    #         warped_points_ransac = PointsToPixelCoords(warped_points_ransac_norm, source_im_size)
    #         #print(warped_points_ransac.data.numpy()[0].reshape((12,2)))
    #         #print(source_points_np)
    if do_aff:
        theta_aff,correlationAB,correlationBA =model_aff(batch)
        keypoints_A, keypoints_B = find_mutual_matached_keypoints(correlationAB, correlationBA)
        print('Mutual Keypoints: ',keypoints_A.shape[0])
        # do affine only
        warped_points_aff_norm = pt.affPointTnf(theta_aff,target_points_norm)
        warped_points_aff = PointsToPixelCoords(warped_points_aff_norm,source_im_size)
        if do_ransac:
            keypoints_A, keypoints_B = find_mutual_matached_keypoints(correlationAB, correlationBA)

            #keypoints_A = keypoints_A.reshape((1,2,-1))
            #keypoints_B = keypoints_B.reshape((1,2, -1))
            tensor_shape = correlationAB.data.numpy()[0].shape
            #print(keypoints_A)
            #print(keypoints_B)

            im_keypointsA = tensorPointstoPixels(keypoints_A,tensor_size=tensor_shape,im_size=(source_im_shape_np[1],source_im_shape_np[0]))
            im_keypointsB = tensorPointstoPixels(keypoints_B, tensor_size=tensor_shape,
                                                 im_size=(target_im_shape_np[1],target_im_shape_np[0]))
            #print(keypoints_A)
            #print (im_keypointsA)
            #break
            # tensor_size = Variable(torch.Tensor(np.asarray((tensor_shape[1],tensor_shape[2],tensor_shape[0])).reshape(1,-1).astype(np.float32)))
            # keyptsA_tensor = Variable(torch.Tensor(keypoints_A.astype(np.float32)))
            # keyptsB_tensor = Variable(torch.Tensor(keypoints_B.astype(np.float32)))
            # keyptsA_tensor_var = PointsToUnitCoords(keyptsA_tensor, tensor_size)
            # keyptsB_tensor_var = PointsToUnitCoords(keyptsB_tensor, tensor_size)
            M = estimateAffineRansac(im_keypointsB,im_keypointsA)
            ransac_success = M is not None
            if ransac_success:
                M_var = Variable(torch.FloatTensor(M.flatten()))
                # do affine only
                warped_points_ransac_norm = pt.affPointTnf(M_var, target_points)
                warped_points_ransac = warped_points_ransac_norm#PointsToPixelCoords(warped_points_ransac_norm, source_im_size)

    if do_tps:
        theta_tps,_,_=model_tps(batch)
        
        # do tps only
        warped_points_tps_norm = pt.tpsPointTnf(theta_tps,target_points_norm)
        warped_points_tps = PointsToPixelCoords(warped_points_tps_norm,source_im_size)
        
    if do_aff and do_tps:
        warped_image_aff = affTnf(batch['source_image'],theta_aff.view(-1,2,3))
        theta_aff_tps,_,_=model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})

        # do tps+affine
        warped_points_aff_tps_norm = pt.tpsPointTnf(theta_aff_tps,target_points_norm)
        warped_points_aff_tps_norm = pt.affPointTnf(theta_aff,warped_points_aff_tps_norm)
        warped_points_aff_tps = PointsToPixelCoords(warped_points_aff_tps_norm,source_im_size)
    
    L_pck = batch['L_pck'].data

    if do_ransac and ransac_success:
        correct_points_ransac, num_points,reprojection_error = correct_keypoints(source_points.data,
                                                           warped_points_ransac.data, L_pck)
        print(correct_points_ransac)
        total_correct_points_ransac += correct_points_ransac

    if do_aff:
        correct_points_aff, num_points,reprojection_error = correct_keypoints(source_points.data,
                                                       warped_points_aff.data,L_pck)
        print('Reprojection error: ', reprojection_error)
        total_correct_points_aff += correct_points_aff
        
    if do_tps:
        correct_points_tps, num_points,reprojection_error = correct_keypoints(source_points.data,
                                                           warped_points_tps.data,L_pck)
        total_correct_points_tps += correct_points_tps

    if do_aff and do_tps:
        correct_points_aff_tps, num_points,reprojection_error = correct_keypoints(source_points.data,
                                                           warped_points_aff_tps.data,L_pck)
        total_correct_points_aff_tps += correct_points_aff_tps        

    total_points += num_points
    print("============")
    print('Batch: [{}/{} ({:.0f}%)]'.format(i, len(dataloader), 100. * i / len(dataloader)))

if do_ransac:
    PCK_ransac=total_correct_points_ransac/total_points
    print('PCK ransac:',PCK_ransac)
if do_aff:
    PCK_aff=total_correct_points_aff/total_points
    print('PCK affine:',PCK_aff)
if do_tps:
    PCK_tps=total_correct_points_tps/total_points
    print('PCK tps:',PCK_tps)
if do_aff and do_tps:
    PCK_aff_tps=total_correct_points_aff_tps/total_points
    print('PCK affine+tps:',PCK_aff_tps)
print('Done!')



