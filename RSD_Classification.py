import pandas as pd
from classifier import *
import cv2

b = 'runs/train/exp8/weights/best.pt'
p = 'GTSRB/Images'


def classify_sign(img):
    cls = torch.load(b, map_location=torch.device('cpu'))['model'].float()
    tup = classify2(cls, size=128, file=img)
    classID = tup.argmax().item()
    return classID