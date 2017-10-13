import os
import cv2
import argparse
import pandas as pd
import csv
import string
import numpy as np
WIDTH,HEIGHT = 240,240
# Argument parsing
parser = argparse.ArgumentParser(description='Keypoints Generation for CNN Geometric Matching')

# Paths
parser.add_argument('--path', type=str, default='datasets/PF-dataset', help='Path to PF dataset')
parser.add_argument('--pairs', type=str, default='Pairs CSV file', help='Path to PF dataset')
parser.add_argument('--pts', type=int, default=12, help='Number of Keypoints')
args = parser.parse_args()
dataset_base_dir = args.path
pairs_file = args.pairs

#fast_detector = cv2.FastFeatureDetector_create(threshold=30)

# Initiate SIFT detector


#feature_detector = cv2.ORB_create()#cv2.xfeatures2d.SIFT_create()
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

pd_data_frame = pd.read_csv(pairs_file)

min_num_kps = args.pts
df_columns=['imageA','imageB']
df_columns.extend(['XA'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['YA'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['XB'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['YB'+str(i) for i in range(1,min_num_kps+1)])

output_keypoints_file = string.replace(pairs_file,'.csv','_sift_RANSAC_{0}kps.csv'.format(min_num_kps))
out_data_frame = pd.DataFrame(columns=df_columns,)
for index,row in pd_data_frame.iterrows():
    imgA_filename = row[0]
    imgB_filename = row[1]
    print(imgA_filename)
    imgA = cv2.imread(os.path.join(dataset_base_dir,imgA_filename),0)
    imgB = cv2.imread(os.path.join(dataset_base_dir, imgB_filename), 0)
    #resized_imgA = cv2.resize(imgA,(WIDTH,HEIGHT))
    #resized_imgB = cv2.resize(imgB, (WIDTH, HEIGHT))
    #final_kpsA, final_kpsB , _ = generate_matched_keypoints(feature_detector,imgA,imgB)
    kpsA,kpsB ,_ = generate_matched_keypoints(feature_detector,imgA,imgB)
    ptsA = np.int32(kpsA)
    ptsB = np.int32(kpsB)
    F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC)

    final_kpsA = ptsA[mask.ravel() == 1]
    final_kpsB = ptsB[mask.ravel() == 1]
    if len(final_kpsA) < min_num_kps or len(final_kpsB) < min_num_kps :
        continue
    row = []
    row.append(imgA_filename)
    row.append(imgB_filename)
    row.extend([kp[0] for kp in final_kpsA[:min_num_kps]])
    row.extend([kp[1] for kp in final_kpsA[:min_num_kps]])
    row.extend([kp[0] for kp in final_kpsB[:min_num_kps]])
    row.extend([kp[1] for kp in final_kpsB[:min_num_kps]])
    print(len(row))
    #df = pd.DataFrame(data=[row],columns=df_columns)
    out_data_frame.loc[out_data_frame.shape[0]] = row

out_data_frame.to_csv(output_keypoints_file,index=False)
print('Done')
    #img_matches = cv2.drawMatches(resized_imgA, final_kpsA, resized_imgB, final_kpsB, matches[:10], flags=2,outImg=None)
    #cv2.imshow("Matches",img_matches)
    #cv2.waitKey(500)