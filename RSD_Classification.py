import pandas as pd
from classifier import *
import cv2

b = 'runs/train/exp8/weights/best.pt'
p = 'GTSRB/Images'


def classify_sign(best, im_list):
    l = []
    cls = torch.load(best, map_location=torch.device('cpu'))['model'].float()
    # files = Path(p).glob('*.jpg')  # images from dir
    for f in im_list:  # first 10 images
        tup = classify2(cls, size=128, file=f)
        classID = tup.argmax().item()
        l.append(classID)
    return l


def process_img(img, l):
    img_list = []
    for i, _ in enumerate(l):
        if l[i][5] == 1:
            x1 = int(l[i][0])
            y1 = int(l[i][1])
            x2 = int(l[i][2])
            y2 = int(l[i][3])
            cropped_image = img[y1:y2, x1:x2]
            img_list.append(cropped_image)
            # cv2.imshow(f"crop{i}", cropped_image)
    classID = classify_sign(b, img_list)
    j = 0
    for i, _ in enumerate(l):
        if l[i][5] == 1:
            l[i].append(classID[j])
            j += 1
        else:
            l[i].append(0)
    return l
