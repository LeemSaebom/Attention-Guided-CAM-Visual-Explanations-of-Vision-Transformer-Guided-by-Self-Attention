import numpy as np
import cv2
import torch

def getBoudingBox_multi(mask, threshold=None):    
    mask = mask.detach().cpu().numpy()[0]
    mask = np.transpose(mask, (1, 2, 0))
    if threshold==None:
        threshold = mask.mean()
    mask[mask>threshold]=1
    mask[mask<=threshold]=0
    mask = (mask*255).astype(np.uint8)
    # ret, thr = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bnd_box=torch.tensor([])
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        xmin = x
        ymin = y
        xmax = x+w
        ymax = y+h
        if bnd_box.dim()==1:
            bnd_box = torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)
        else:
            bnd_box = torch.cat((bnd_box, torch.tensor((xmin, ymin, xmax, ymax)).unsqueeze(0)), dim=0)
    return bnd_box



def box_to_seg(box_cor):
    segmask = torch.zeros((224, 224))
    if box_cor.dim()!=1:
        n, _ = box_cor.shape
        for i in range(n):
            xmin = box_cor[i][0]
            ymin = box_cor[i][1]
            xmax = box_cor[i][2]
            ymax = box_cor[i][3]
            segmask[ymin:ymax+1, xmin:xmax+1]=1
    
    return segmask