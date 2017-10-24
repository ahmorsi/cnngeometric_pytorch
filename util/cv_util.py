import cv2
import numpy as np
def find_mutual_matached_keypoints(correlationAB,correlationBA):
    np_corAB = correlationAB.data.numpy()[0]
    np_corBA = correlationBA.data.numpy()[0]
    d,w,h = np_corAB.shape
    np_corAB = np_corAB.reshape((w,h,d))
    np_corBA = np_corBA.reshape((w, h, d))

    keypoints_A = []
    keypoints_B = []
    for i in range(w):
        for j in range(h):
            best_A_idx = np.argmax(np_corAB[i,j])
            row = int(best_A_idx / w)
            col = int(best_A_idx % w)
            best_B_idx = np.argmax(np_corBA[col,row])
            curB_idx = h*(j-1) + i
            if curB_idx != best_B_idx:
                continue

            keypoints_B.append((j, i))
            keypoints_A.append((row,col))
    return np.float32(keypoints_A),np.float32(keypoints_B)
def estimateAffineRansac(keypoints_A, keypoints_B):
    if len(keypoints_A) == 0:
        return None
    ptsA = np.int32(keypoints_A)
    ptsB = np.int32(keypoints_B)
    M = cv2.estimateRigidTransform(ptsB, ptsA, fullAffine=True)
    return M
def tensorPointstoPixels(points,tensor_size,im_size):
    img_width ,img_height = im_size
    d,n,m=tensor_size
    ratio_width, ratio_height = img_width // n, img_height // m
    newPts = []
    for x,y in points:
        ptx, pty = (x * ratio_width), (y * ratio_height)
        newPts.append((pty, ptx))
    return np.array(newPts)