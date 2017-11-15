from __future__ import print_function, division
import os
import argparse
import torch
import torch.nn as nn
from model.cnn_geometric_model import CNNGeometric,CNNGeometricRegression,FeatureExtraction
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars, readImage, correct_keypoints, compute_reprojection_error
from util.cv_util import find_mutual_matached_keypoints, tensorPointstoPixels
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable
import numpy as np

print('CNN Geometric Verification script')

output_size = (240, 240)
_use_cuda = torch.cuda.is_available()
__transform = NormalizeImageDict(['source_image', 'target_image'])
__batchTensorToVars = BatchTensorToVars(use_cuda=_use_cuda)

class CNNGeometricMatcher:
    def __init__(self, use_extracted_features = False, geometric_affine_model=None,geometric_tps_model=None,arch='resnet18',featext_weights=None, min_mutual_keypoints=6,
                 min_reprojection_error=200):
        self.min_mutual_keypoints = min_mutual_keypoints
        self.min_reprojection_error = min_reprojection_error
        self.__do_affine = geometric_affine_model is not None
        self.__do_tps = not use_extracted_features and geometric_tps_model is not None
        self.__affTnf = GeometricTnf(geometric_model='affine',use_cuda= _use_cuda )
        if self.__do_affine:
            checkpoint = torch.load(geometric_affine_model, map_location=lambda storage, loc: storage)
            print('Loading CNN Affine Geometric Model')
            if use_extracted_features:
                self.model_affine = CNNGeometricRegression(use_cuda=use_cuda, geometric_model='affine', arch=arch,
                                             featext_weights=featext_weights)
                model_dict = self.model_affine.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
                model_dict.update(pretrained_dict)
                self.model_affine.load_state_dict(model_dict)
            else:
                self.model_affine = CNNGeometric(use_cuda= _use_cuda , geometric_model='affine', arch=arch,
                                             featext_weights=featext_weights)

                self.model_affine.load_state_dict(checkpoint['state_dict'])

            self.model_affine.eval()

        if self.__do_tps:
            self.model_tps = CNNGeometric(use_cuda=use_cuda, geometric_model='tps',arch=arch,featext_weights=featext_weights)
            checkpoint = torch.load(geometric_tps_model, map_location=lambda storage, loc: storage)
            print('Loading CNN TPS Geometric Model')
            #self.model_tps.load_state_dict(checkpoint['state_dict'])
            model_dict = self.model_tps.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.model_tps.load_state_dict(model_dict)
            self.model_tps.eval()

        self.pt = PointTnf(use_cuda=_use_cuda)

    def run(self, batch):
        if self.__do_affine:
            theta_aff, correlationAB, correlationBA = self.model_affine(batch)

        if self.__do_tps:
            if self.__do_affine:
                warped_image_aff = self.__affTnf(batch['source_image'], theta_aff.view(-1, 2, 3))
                theta_affine_tps, correlationAB, correlationBA = self.model_tps({'source_image': warped_image_aff, 'target_image': batch['target_image']})
            else:
                theta_tps, correlationAB, correlationBA = self.model_tps(batch)

        keypoints_A, keypoints_B = find_mutual_matached_keypoints(correlationAB, correlationBA)
        num_mutual_keypoints = keypoints_A.shape[0]

        if num_mutual_keypoints < self.min_mutual_keypoints:
            matched = False
            reprojection_error = -1
        else:
            source_im_size = batch['source_im_size']
            target_im_size = batch['target_im_size']

            source_im_shape_np = source_im_size.data.numpy()
            target_im_shape_np = target_im_size.data.numpy()

            tensor_shape = correlationAB.data.numpy()[0].shape

            im_keypointsA = tensorPointstoPixels(keypoints_A, tensor_size=tensor_shape,
                                                 im_size=(source_im_shape_np[0][1], source_im_shape_np[0][0]))
            im_keypointsB = tensorPointstoPixels(keypoints_B, tensor_size=tensor_shape,
                                                 im_size=(target_im_shape_np[0][1], target_im_shape_np[0][0]))

            torch_keypointsA_var = Variable(torch.Tensor(im_keypointsA.reshape(1, 2, -1).astype(np.float32)))
            torch_keypointsB_var = Variable(torch.Tensor(im_keypointsB.reshape(1, 2, -1).astype(np.float32)))


            target_points_norm = PointsToUnitCoords(torch_keypointsB_var, target_im_size)

            if self.__do_affine and self.__do_tps:
                warped_points_aff_tps_norm = self.pt.tpsPointTnf(theta_affine_tps, target_points_norm)
                warped_points_norm = self.pt.affPointTnf(theta_aff, warped_points_aff_tps_norm)
            elif self.__do_affine:
                warped_points_norm = self.pt.affPointTnf(theta_aff, target_points_norm)
            elif self.__do_tps:
                warped_points_norm = self.pt.tpsPointTnf(theta_tps, target_points_norm)

            warped_points_aff = PointsToPixelCoords(warped_points_norm, source_im_size)
            reprojection_error = compute_reprojection_error(torch_keypointsA_var, warped_points_aff)
            matched = reprojection_error <= self.min_reprojection_error
        return reprojection_error, matched, num_mutual_keypoints


def read_input(path_imgA, path_imgB, output_size=(240, 240)):
    out_h, out_w = output_size
    resizeTnf = GeometricTnf(out_h=out_h, out_w=out_w, use_cuda=False)
    image_A, im_size_A = readImage(img_path=path_imgA, affineTnf=resizeTnf)
    image_B, im_size_B = readImage(img_path=path_imgB, affineTnf=resizeTnf)

    sample = {'source_image': image_A, 'target_image': image_B, 'source_im_size': im_size_A,
              'target_im_size': im_size_B}

    sample = __transform(sample)

    batch = __batchTensorToVars(sample)
    return batch

def read_features_input(path_tensorA, path_tensorB,image_shape):
    src_features_tensor = torch.load(path_tensorA)
    target_features_tensor = torch.load(path_tensorB)

    im_size = np.asarray(image_shape)
    im_size = torch.Tensor(im_size.reshape(1, -1).astype(np.float32))

    sample = {'source_features':src_features_tensor,'target_features':target_features_tensor,
              'source_im_size': im_size,'target_im_size': im_size
              }
    batch = __batchTensorToVars(sample)
    return batch

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='CNNGeometric PyTorch implementation')
    # Paths
    parser.add_argument('--model-aff', type=str,
                        default='trained_models/best_streetview_checkpoint_adam_affine_grid_loss.pth.tar',
                        help='Trained affine model filename')
    parser.add_argument('--model-tps', type=str,
                        default='trained_models/best_streetview_checkpoint_adam_tps_grid_loss.pth.tar',
                        help='Trained TPS model filename')
    parser.add_argument('--imgA', type=str,
                        default='/home/develop/Work/Datasets/GardensPointWalking/day_left/Image005.jpg',
                        help='Path to Source Image')
    parser.add_argument('--imgB', type=str,
                        default='/home/develop/Work/Datasets/GardensPointWalking/day_right/Image005.jpg',
                        help='Path to Target Image')
    args = parser.parse_args()

    FeatureExtraction = FeatureExtraction(use_cuda=__use_cuda, arch='vgg16')
    batch = read_input(args.imgA,args.imgB)

    cnn_regressor = CNNGeometricRegression(use_cuda=__use_cuda, geometric_model='affine')
    checkpoint = torch.load(args.model_aff, map_location=lambda storage, loc: storage)

    model_dict = cnn_regressor.state_dict()
    pretrained_dict = {k:v for k,v in checkpoint['state_dict'].items() if k in model_dict}
    model_dict.update(pretrained_dict)
    cnn_regressor.load_state_dict(model_dict)
    print('Loaded Model')

    src_np = np.random.rand(1,512,15,15)
    target_np = np.random.rand(1,512,15, 15)
    src_tensor = FeatureExtraction(batch['source_image'])#Variable(torch.Tensor(src_np))
    target_tensor = FeatureExtraction(batch['target_image'])#Variable(torch.Tensor(target_np))
    torch.save(src_tensor,'0.pt')
    torch.save(target_tensor, '1.pt')

    loaded_src_tensor = torch.load('0.pt')
    loaded_target_tensor = torch.load('1.pt')
    batch = {'source_features':loaded_src_tensor,'target_features':loaded_target_tensor}
    theta_aff = cnn_regressor(batch)
    print('Done')
    #
    # matcher = CNNGeometricMatcher(use_cuda=__use_cuda, geometric_affine_model=args.model_aff,
    #                               geometric_tps_model=None,arch='vgg16',
    #                               min_mutual_keypoints=4,min_reprojection_error=150)
    #
    # batch = read_input(path_imgA=args.imgA, path_imgB=args.imgB)
    # reprojection_error, matched, num_mutual_keypoints = matcher.run(batch)
    # print('Num of Mutual Keypoints: ', num_mutual_keypoints)
    # print('Geometric Matched: ', matched)
    # print('Reprojection Error: ', reprojection_error)
