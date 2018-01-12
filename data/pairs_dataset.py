from __future__ import print_function, division
import torch
import os
from skimage import io
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from geotnf.transformation import GeometricTnf
from torch.autograd import Variable


class PairsDataset(Dataset):
    """

    Synthetically transformed pairs dataset for training with strong supervision

    Args:
            csv_file (string): Path to the csv file with image names and transformations.
            training_image_path (string): Directory with all the images.
            transform (callable): Transformation for post-processing the training pair (eg. image normalization)

    Returns:
            Dict: {'image': full dataset image, 'theta': desired transformation}

    """

    def __init__(self, csv_file, training_image_path, geometric_model='pose',use_timestamp=False,output_size=(480, 640), transform=None):
        self.out_h, self.out_w = output_size
        # read csv file
        self.train_data = pd.read_csv(csv_file)
        self.img_src_names = self.train_data.iloc[:, 0]
        self.img_target_names = self.train_data.iloc[:, 1]
        self.use_timestamp = use_timestamp
        if self.use_timestamp:
            self.timestamps = self.train_data.iloc[:,2]
        # copy arguments
        self.training_image_path = training_image_path
        self.transform = transform
        self.geometric_model = geometric_model
        self.affineTnf = GeometricTnf(out_h=self.out_h, out_w=self.out_w, use_cuda=False)

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        # read image
        img_src_name = os.path.join(self.training_image_path, self.img_src_names[idx])
        image_src = io.imread(img_src_name)

        img_target_name = os.path.join(self.training_image_path, self.img_target_names[idx])
        image_target = io.imread(img_target_name)

        # make arrays float tensor for subsequent processing
        image_src = torch.Tensor(image_src.astype(np.float32))
        image_target = torch.Tensor(image_target.astype(np.float32))

        # permute order of image to CHW
        image_src = image_src.transpose(1, 2).transpose(0, 1)
        image_target = image_target.transpose(1, 2).transpose(0, 1)

        # Resize image using bilinear sampling with identity affine tnf
        if image_src.size()[0] != self.out_h or image_src.size()[1] != self.out_w:
            image_src = self.affineTnf(Variable(image_src.unsqueeze(0), requires_grad=False)).data.squeeze(0)

        if image_target.size()[0] != self.out_h or image_target.size()[1] != self.out_w:
            image_target = self.affineTnf(Variable(image_target.unsqueeze(0), requires_grad=False)).data.squeeze(0)


        sample = {'source_image': image_src, 'target_image': image_target}

        if self.transform:
            sample = self.transform(sample)

        if self.use_timestamp:
            sample['timestamp'] = self.timestamps[idx]

        return img_src_name,img_target_name,sample