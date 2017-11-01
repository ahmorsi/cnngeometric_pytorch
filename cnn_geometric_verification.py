from __future__ import print_function, division
import os
import argparse
import torch
import torch.nn as nn
from model.cnn_geometric_model import CNNGeometric
from image.normalization import NormalizeImageDict
from util.torch_util import BatchTensorToVars, readImage, correct_keypoints, compute_reprojection_error
from util.cv_util import find_mutual_matached_keypoints, tensorPointstoPixels
from geotnf.point_tnf import *
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable

print('CNN Geometric Verification script')

output_size = (240, 240)


class CNNGeometricMatcher:
    def __init__(self, model_path=None, use_cuda=True, geometric_model='affine', min_mutual_keypoints=6,
                 min_reprojection_error=200):
        self.min_mutual_keypoints = min_mutual_keypoints
        self.min_reprojection_error = min_reprojection_error
        self.model = CNNGeometric(use_cuda=use_cuda, geometric_model=geometric_model)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        if model_path is not None:
            print('Loading CNN Geometric Model')
            self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.pt = PointTnf(use_cuda=use_cuda)

    def run(self, batch):
        theta_aff, correlationAB, correlationBA = self.model(batch)
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

            warped_points_aff_norm = self.pt.affPointTnf(theta_aff, target_points_norm)
            warped_points_aff = PointsToPixelCoords(warped_points_aff_norm, source_im_size)
            reprojection_error = compute_reprojection_error(torch_keypointsA_var, warped_points_aff)
            matched = reprojection_error <= self.min_reprojection_error
        return reprojection_error, matched, num_mutual_keypoints


__use_cuda = torch.cuda.is_available()
__transform = NormalizeImageDict(['source_image', 'target_image'])
__batchTensorToVars = BatchTensorToVars(use_cuda=__use_cuda)


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
                        default='/home/develop/Work/Datasets/GardensPointWalking/day_right/Image001.jpg',
                        help='Path to Target Image')
    args = parser.parse_args()

    matcher = CNNGeometricMatcher(use_cuda=__use_cuda, geometric_model='affine', model_path=args.model_aff,
                                  min_mutual_keypoints=4,min_reprojection_error=150)

    batch = read_input(path_imgA=args.imgA, path_imgB=args.imgB)
    reprojection_error, matched, num_mutual_keypoints = matcher.run(batch)
    print('Num of Mutual Keypoints: ', num_mutual_keypoints)
    print('Geometric Matched: ', matched)
    print('Reprojection Error: ', reprojection_error)
