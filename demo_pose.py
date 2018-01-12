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
from image.normalization import NormalizeImageDict, normalize_image
from util.torch_util import BatchTensorToVars, str_to_bool
from geotnf.transformation import GeometricTnf
from geotnf.point_tnf import *
import matplotlib.pyplot as plt
from skimage import io,color
import warnings
import cv2
warnings.filterwarnings('ignore')

# for compatibility with Python 2
try:
    input = raw_input
except NameError:
    pass

def calculateEssentialMatrix(R,T):
    T_matrix = np.array([[0,-T[2,0],T[1,0]],
                         [T[2,0],0,-T[0,0]],
                         [-T[1,0],T[0,0],0]])
    E = R*T_matrix
    return E

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    row,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, (-r[2]/r[1])%row ])
        x1,y1 = map(int, [c, (-(r[2]+r[0]*c)/r[1])%row ])
        cv2.line(img1, (x0,y0), (x1,y1), color,1)
        cv2.circle(img1,tuple(pt1),5,color,-1)
        cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

feature_detector = cv2.xfeatures2d.SIFT_create()
def generate_matched_keypoints(feature_detector,imgA,imgB):
    (kpsA, descsA) = feature_detector.detectAndCompute(imgA, None)
    (kpsB, descsB) = feature_detector.detectAndCompute(imgB, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descsA, descsB, k=2)
    # Match descriptors.
    #good = bf.match(descsA, descsB)
    # Apply ratio test
    good = []
    for m, n in matches:
         if m.distance < 0.8 * n.distance:
             good.append(m)
    final_kpsA = []
    final_kpsB = []

    for m in sorted(good, key = lambda x:x.distance):
        kpBIdx = m.trainIdx
        kpAIdx = m.queryIdx
        pntA = kpsA[kpAIdx]
        pntB = kpsB[kpBIdx]
        final_kpsA.append(pntA.pt)
        final_kpsB.append(pntB.pt)
    #return kpsA,kpsB,good
    return final_kpsA, final_kpsB,good

print('CNNGeometric PF demo script')
use_cuda = torch.cuda.is_available()
# Argument parsing
parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
# Paths
parser.add_argument('--model', type=str, default='trained_models/best_checkpoint_resnet18_adam_pose_mse_loss.pth.tar', help='Trained affine model filename')
#parser.add_argument('--model-tps', type=str, default='trained_models/best_pascal_checkpoint_adam_tps_grid_loss.pth.tar', help='Trained TPS model filename')
parser.add_argument('--path', type=str, default='/home/develop/Work/Datasets/', help='Path to PF dataset')
parser.add_argument('--pairs', type=str, default='/home/develop/Work/Datasets/gardens_pairs_path_samples_sift_RANSAC_12kps.csv', help='Path to PF dataset')
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
dataset = PlacesDataset(csv_file=dataset_pairs_file,
                    training_image_path=dataset_path,
                    transform=NormalizeImageDict(['source_image','target_image']))
dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=4)
batchTensorToVars = BatchTensorToVars(use_cuda=use_cuda)

for source_im_path,target_im_path,batch in dataloader:
    # get random batch of size 1
    batch = batchTensorToVars(batch)

    source_im_size = batch['source_im_size']
    target_im_size = batch['target_im_size']

    source_points = batch['source_points']
    target_points = batch['target_points']

    model.eval()
    theta_pose,_,_ = model(batch)

    #source_im_path = batch['source_im_path']
    #target_im_path = batch['target_im_path']

    data_np = theta_pose.data.numpy()
    R = data_np[0,0:9].reshape(3,3)
    T = data_np[0,9:].reshape(3,1)
    E = calculateEssentialMatrix(R,T)
    det = np.linalg.det(E)
    if det == 0:
        print('invalid E')
        continue
    F = E
    img1 = cv2.imread(source_im_path[0],0)
    img2 = cv2.imread(target_im_path[0],0)

    kpsA,kpsB ,_ = generate_matched_keypoints(feature_detector,img1,img2)
    pts1 = np.int32(kpsA)[:12]
    pts2 = np.int32(kpsB)[:12]

    #pts1, pts2 = source_points.data.numpy()[0].reshape(-1,2), target_points.data.numpy()[0].reshape(-1,2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2, 2, F)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1, 1, F)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    plt.subplot(121), plt.imshow(img5)
    plt.subplot(122), plt.imshow(img3)
    plt.show()
    #cv2.imshow('left',img5)
    #cv2.imshow('right', img3)
    #cv2.waitKey(300)
    #res = input('Run for another example ([y]/n): ')
    #if res == 'n':
    #    break
