from __future__ import print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from util import torch_util

class FeatureExtraction(torch.nn.Module):
    def __init__(self, use_cuda=True,arch='resnet18',weights=None):
        super(FeatureExtraction, self).__init__()

        if arch == 'vgg16':
            self.model = models.vgg16(pretrained=True)
            # keep feature extraction network up to pool4 (last layer - 7)
            self.model = nn.Sequential(*list(self.model.features.children())[:-7])
        elif arch == 'resnet18':
            imageNet = weights is None
            self.model = models.resnet18(pretrained=imageNet)
            if weights is not None:
                self.model.load_state_dict(torch_util.convertGpuWeightsToCpu(weights))
            self.model = nn.Sequential(*list(self.model.children())[:-3])
        # freeze parameters
        for param in self.model.parameters():
            param.requires_grad = False
        # move to GPU
        if use_cuda:
            self.model.cuda()
        
    def forward(self, image_batch):
        return self.model(image_batch)
    
class FeatureL2Norm(torch.nn.Module):
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature):
        epsilon = 1e-6
#        print(feature.size())
#        print(torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).size())
        norm = torch.pow(torch.sum(torch.pow(feature,2),1)+epsilon,0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature,norm)
    
class FeatureCorrelation(torch.nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()
    
    def forward(self, feature_A, feature_B):
        b,c,h,w = feature_A.size()
        # reshape features for matrix multiplication
        feature_A = feature_A.transpose(2,3).contiguous().view(b,c,h*w)
        feature_B = feature_B.view(b,c,h*w).transpose(1,2)
        # perform matrix mult.
        feature_mul = torch.bmm(feature_B,feature_A)
        correlation_tensor = feature_mul.view(b,h,w,h*w).transpose(2,3).transpose(1,2)
        return correlation_tensor
    
class FeatureRegression(nn.Module):
    def __init__(self, input_dim=225,output_dim=6, use_cuda=True):
        super(FeatureRegression, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 128, kernel_size=7, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.linear = nn.Linear(64 * 5 * 5, output_dim)
        if use_cuda:
            self.conv.cuda()
            self.linear.cuda()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

class CNNGeometric(nn.Module):
    def __init__(self, geometric_model='affine',arch='resnet18',featext_weights=None, normalize_features=True, normalize_matches=True, batch_normalization=True, use_cuda=True):
        super(CNNGeometric, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        self.FeatureExtraction = FeatureExtraction(use_cuda=self.use_cuda,arch=arch,weights=featext_weights)
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        if geometric_model=='affine':
            output_dim = 6
        elif geometric_model=='tps':
            output_dim = 18
        elif geometric_model == 'pose':
            output_dim = 7

        if arch == 'resnet18':
            input_dim = 225
        elif arch == 'vgg16':
            input_dim = 225

        self.FeatureRegression = FeatureRegression(input_dim=input_dim,output_dim=output_dim,use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, tnf_batch):
        # do feature extraction
        feature_A = self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = self.FeatureExtraction(tnf_batch['target_image'])

        # normalize
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        correlation_BA = self.FeatureCorrelation(feature_B, feature_A)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
            correlation_BA = self.FeatureL2Norm(self.ReLU(correlation_BA))
#        correlation = self.FeatureL2Norm(correlation)
        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        return theta,correlation,correlation_BA

class CNNGeometricRegression(nn.Module):
    def __init__(self, geometric_model='affine',arch='resnet18',featext_weights=None, normalize_features=True, normalize_matches=True, batch_normalization=True, use_cuda=True):
        super(CNNGeometricRegression, self).__init__()
        self.use_cuda = use_cuda
        self.normalize_features = normalize_features
        self.normalize_matches = normalize_matches
        #self.FeatureExtraction = FeatureExtraction(use_cuda=self.use_cuda,arch=arch,weights=featext_weights)
        self.FeatureL2Norm = FeatureL2Norm()
        self.FeatureCorrelation = FeatureCorrelation()
        if geometric_model=='affine':
            output_dim = 6
        elif geometric_model=='tps':
            output_dim = 18
        elif geometric_model == 'pose':
            output_dim = 12

        if arch == 'resnet18':
            input_dim = 225
        elif arch == 'vgg16':
            input_dim = 225

        self.FeatureRegression = FeatureRegression(input_dim=input_dim,output_dim=output_dim,use_cuda=self.use_cuda)
        self.ReLU = nn.ReLU(inplace=True)

    def forward(self, tnf_batch):
        # do feature extraction
        feature_A = tnf_batch['source_features']#self.FeatureExtraction(tnf_batch['source_image'])
        feature_B = tnf_batch['target_features']#self.FeatureExtraction(tnf_batch['target_image'])
        # normalize
        if self.normalize_features:
            feature_A = self.FeatureL2Norm(feature_A)
            feature_B = self.FeatureL2Norm(feature_B)
        # do feature correlation
        correlation = self.FeatureCorrelation(feature_A,feature_B)
        correlation_BA = self.FeatureCorrelation(feature_B, feature_A)
        # normalize
        if self.normalize_matches:
            correlation = self.FeatureL2Norm(self.ReLU(correlation))
            correlation_BA = self.FeatureL2Norm(self.ReLU(correlation_BA))
#        correlation = self.FeatureL2Norm(correlation)
        # do regression to tnf parameters theta
        theta = self.FeatureRegression(correlation)

        return theta,correlation,correlation_BA