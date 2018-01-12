import os
import csv

def generate_pairs_TUM(filename,img_list,img_folder):
    with open(filename,'w') as csvfile:
        fieldnames = ['source_image','target_image','timestamp']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(len(img_list)-1):
            src_idx = idx
            target_idx = idx + 1
            src_img_name = img_list[src_idx]
            target_img_name = img_list[target_idx]
            src_path = os.path.join(img_folder,src_img_name)
            target_path = os.path.join(img_folder, target_img_name)
            timestamp = target_img_name[:-4]
            writer.writerow({fieldnames[0]: src_path, fieldnames[1]: target_path,fieldnames[2]: float(timestamp)})
            print(src_path, target_path)

#basedir = "/media/develop/Ahmed-HD/Thesis/KITTI/dataset/sequences"
# for x in range(22):
#     seq_dir = os.path.join(basedir,"%02d" % x)
#     images_dir = os.path.join(seq_dir,"image_2")
#     img_list = sorted(os.listdir(images_dir))
#     with open(os.path.join(basedir,"%02d_pairs.csv" % x),'w') as csvfile:
#         fieldnames = ['source_image','target_image']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#         writer.writeheader()
#         for idx in range(len(img_list)-1):
#             src_idx = idx
#             target_idx = idx + 1
#             src_img_name = img_list[src_idx]
#             target_img_name = img_list[target_idx]
#             src_path = os.path.join(os.path.join("%02d" % x,"image_2"),src_img_name)
#             target_path = os.path.join(os.path.join("%02d" % x,"image_2"), target_img_name)
#             writer.writerow({fieldnames[0]: src_path, fieldnames[1]: target_path})
#             print(src_path, target_path)

basedir = "/media/develop/Elements/aba4hi/Dataset/RGBD_Tracking/rgbd_dataset_freiburg2_pioneer_360"
img_folder = "rgb"
images_dir = os.path.join(basedir,img_folder)
img_list = sorted(os.listdir(images_dir))
generate_pairs_TUM(filename=os.path.join(basedir,"seq_pairs.csv"),img_list=img_list,img_folder=img_folder)
