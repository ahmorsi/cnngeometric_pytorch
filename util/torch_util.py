import shutil
import torch
from torch.autograd import Variable
from os import makedirs, remove
from os.path import exists, join, basename, dirname
import numpy as np
from skimage import io

class BatchTensorToVars(object):
    """Convert tensors in dict batch to vars
    """
    def __init__(self, use_cuda=True):
        self.use_cuda=use_cuda
        
    def __call__(self, batch):
        batch_var = {}
        for key,value in batch.items():
            batch_var[key] = Variable(value,requires_grad=False)
            if self.use_cuda:
                batch_var[key] = batch_var[key].cuda()
            
        return batch_var
    
def save_checkpoint(state, is_best, file):
    model_dir = dirname(file)
    model_fn = basename(file)
    # make dir if needed (should be non-empty)
    if model_dir!='' and not exists(model_dir):
        makedirs(model_dir)
    torch.save(state, file)
    if is_best:
        shutil.copyfile(file, join(model_dir,'best_' + model_fn))
        
def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def readImage(img_path, affineTnf):
    image = io.imread(img_path)
    # get image size
    im_size = np.asarray(image.shape)

    # convert to torch Variable
    image = np.expand_dims(image.transpose((2, 0, 1)), 0)
    image = torch.Tensor(image.astype(np.float32))
    image_var = Variable(image, requires_grad=False)

    # Resize image using bilinear sampling with identity affine tnf
    image = affineTnf(image_var).data

    im_size = torch.Tensor(im_size.reshape(1, -1).astype(np.float32))

    return (image, im_size)
# Compute PCK
def correct_keypoints(source_points,warped_points,L_pck,alpha=0.1):
    # compute correct keypoints
    torch_sum = torch.sum(torch.pow(source_points - warped_points, 2), 1)
    point_distance = torch.pow(torch_sum, 0.5).squeeze(1)
    L_pck_mat = L_pck.expand_as(point_distance)
    correct_points = torch.le(point_distance,L_pck_mat*alpha)
    num_of_correct_points = torch.sum(correct_points.data)
    num_of_points = correct_points.numel()
    #temp = torch.sum(torch_sum.data)
    reprojection_error = torch.mean(point_distance.data)#math.mean(temp)
    return (num_of_correct_points,num_of_points,reprojection_error)

# Compute Reprojection Error
def compute_reprojection_error(source_points,warped_points):
    torch_sum = torch.sum(torch.pow(source_points - warped_points, 2), 1)
    point_distance = torch.pow(torch_sum, 0.5).squeeze(1)
    mse = torch.mean(point_distance.data)
    return mse