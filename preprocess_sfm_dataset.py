import numpy as np
import os
import argparse
from pyquaternion import Quaternion
import pandas as pd

def read_image_list(list_file):
    image_list = []
    with open(list_file) as f:
        for line in f:
            tokens = line.split()
            image_name = tokens[0]
            image_list.append(image_name)
    return image_list

def read_camera_indices_mapping(cc_file):
    cc_list = []
    with open(cc_file) as f:
        str_indices = f.readlines()
        cc_list = [int(x) for x in str_indices]
    return cc_list

def gt_generator(gt_file):
    with open(gt_file) as f:
        while True:
            for line in f:
                tokens = line.split()
                c_src_id = int(tokens[0])
                c_target_id = int(tokens[1])
                Rij = [float(x) for x in tokens[2:11]]
                Tij = [float(x) for x in tokens[11:]]
                yield c_src_id,c_target_id,Rij ,Tij
            break
if __name__ == '__main__':
    # Argument parsing
    parser = argparse.ArgumentParser(description='Preprocess SFM Dataset')
    # Paths
    parser.add_argument('--dir', type=str,default="/media/develop/Ahmed-HD/Thesis/dsfm/datasets",
                        help='Dataset Base dir')

    args = parser.parse_args()

    folders = os.listdir(args.dir)
    for f in folders:
        print(f)
        cur_folder = os.path.join(args.dir,f)
        gt_file = os.path.join(cur_folder,"EGs.txt")
        image_list_file = os.path.join(cur_folder,"list.txt")
        cc_file = os.path.join(cur_folder, "cc.txt")
        image_list = read_image_list(image_list_file)
        #cc_mapping = read_camera_indices_mapping(cc_file)
        df_columns = ['imageA', 'imageB','R11','R12','R13','R21','R22','R23','R31','R32','R33','tx','ty','tz']
        out_data_frame = pd.DataFrame(columns=df_columns)
        with open(gt_file) as f1:
            for line in f1:
                tokens = line.split()
                c_src_id = int(tokens[0])
                c_target_id = int(tokens[1])
                Rij = [float(x) for x in tokens[2:11]]
                Tij = [float(x) for x in tokens[11:]]
                src_img_name = image_list[c_src_id]
                target_img_name = image_list[c_target_id]
                #my_q = Quaternion(matrix=Rij)
                row = []
                row.append(os.path.join(f,src_img_name))
                row.append(os.path.join(f, target_img_name))
                #row.append(my_q.real)
                #row.extend(my_q.imaginary.tolist())
                row.extend(Rij)
                row.extend(Tij)
                out_data_frame.loc[out_data_frame.shape[0]] = row

        out_data_frame.to_csv(os.path.join(cur_folder,"gt_relative_pose.csv"), index=False)