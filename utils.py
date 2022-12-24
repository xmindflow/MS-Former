import numpy as np
import cv2


def dice_metric(A, B):
    intersect = np.sum(A * B)
    fsum = np.sum(A)
    ssum = np.sum(B)
    dice = (2 * intersect ) / (fsum + ssum)
    
    return dice    


def hm_metric(A, B):
    intersection = A * B
    union = np.logical_or(A, B)
    hm_score = (np.sum(union) - np.sum(intersection)) / np.sum(union)
    
    return hm_score


def xor_metric(A, GT):
    intersection = A * GT
    union = np.logical_or(A, GT)
    xor_score = (np.sum(union) - np.sum(intersection)) / np.sum(GT)
    
    return xor_score


def create_mask(pred, GT):
    
    kernel = np.ones((7,7),np.uint8) 
    dilated_GT = cv2.dilate(GT, kernel, iterations = 2)

    mult = pred * GT        
    unique, count = np.unique(mult[mult !=0], return_counts=True)
    cls= unique[np.argmax(count)]
    
    lesion = np.where(pred==cls, 1, 0) * dilated_GT
    
    return lesion