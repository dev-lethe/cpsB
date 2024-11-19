import os
import glob
import numpy as np
import cv2
import argparse
from img2feat import CNN
from sklearn.linear_model import LinearRegression

def accuracy(ans, pred):
    suc = 0
    total = 0
    num = len(ans)
    
    for i in range(num):
        jud = abs(pred[i] - ans[i])
        if jud < 0.5:
            suc = suc + 1
        total = total + 1
    
    acc = suc / total

    return acc