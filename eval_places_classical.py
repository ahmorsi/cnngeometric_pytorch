import os
import cv2
import argparse
import pandas as pd
import csv
import string
import numpy as np
from matplotlib import pyplot as plt
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


#feature_detector = cv2.ORB_create()
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

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

def draw_epipolar_lines(imgA,imgB,ptsA,ptsB,F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(ptsB.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    imgA_lines, _ = drawlines(imgA, imgB, lines1, ptsA, ptsB)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(ptsA.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    imgB_lines, _ = drawlines(imgB, imgA, lines2, ptsB, ptsA)
    return imgA_lines,imgB_lines

pd_data_frame = pd.read_csv(pairs_file)

min_num_kps = args.pts
df_columns=['imageA','imageB']
df_columns.extend(['XA'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['YA'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['XB'+str(i) for i in range(1,min_num_kps+1)])
df_columns.extend(['YB'+str(i) for i in range(1,min_num_kps+1)])

output_keypoints_file = string.replace(pairs_file,'.csv','_{0}kps.csv'.format(min_num_kps))
out_data_frame = pd.DataFrame(columns=df_columns)
for index,row in pd_data_frame.iterrows():
    imgA_filename = row[0]
    imgB_filename = row[1]
    print(imgA_filename)
    imgA = cv2.imread(os.path.join(dataset_base_dir,imgA_filename),0)
    imgB = cv2.imread(os.path.join(dataset_base_dir, imgB_filename), 0)
    #resized_imgA = cv2.resize(imgA,(WIDTH,HEIGHT))
    #resized_imgB = cv2.resize(imgB, (WIDTH, HEIGHT))
    final_kpsA, final_kpsB , _ = generate_matched_keypoints(feature_detector,imgA,imgB)
    if len(final_kpsA) < min_num_kps or len(final_kpsB) < min_num_kps :
        continue

    ptsA = np.int32(final_kpsA[:12])
    ptsB = np.int32(final_kpsB[:12])
    F, mask = cv2.findFundamentalMat(ptsA, ptsB, cv2.FM_RANSAC)

    # We select only inlier points
    ptsA = ptsA[mask.ravel() == 1]
    ptsB = ptsB[mask.ravel() == 1]

    print(len(ptsA), len(ptsB))
    imgA_lines, imgB_lines = draw_epipolar_lines(imgA,imgB,ptsA,ptsB,F)
    plt.subplot(121), plt.imshow(imgA_lines)
    plt.subplot(122), plt.imshow(imgB_lines)
    plt.show()
    cv2.waitKey()
    # row = []
    # row.append(imgA_filename)
    # row.append(imgB_filename)
    # row.extend([kp.pt[0] for kp in final_kpsA[:min_num_kps]])
    # row.extend([kp.pt[1] for kp in final_kpsA[:min_num_kps]])
    # row.extend([kp.pt[0] for kp in final_kpsB[:min_num_kps]])
    # row.extend([kp.pt[1] for kp in final_kpsB[:min_num_kps]])
    # print(len(row))
    # #df = pd.DataFrame(data=[row],columns=df_columns)
    # out_data_frame.loc[out_data_frame.shape[0]] = row

#out_data_frame.to_csv(output_keypoints_file,index=False)
print('Done')
    #img_matches = cv2.drawMatches(resized_imgA, final_kpsA, resized_imgB, final_kpsB, matches[:10], flags=2,outImg=None)
    #cv2.imshow("Matches",img_matches)
    #cv2.waitKey(500)