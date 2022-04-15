from classifier import *
import cv2
import numpy as np

b = 'runs/train/exp8/weights/best.pt'
p = 'GTSRB/Images'
t = []
model = torch.load(b, map_location=torch.device('cpu'))['model'].float()
files = Path(p).glob('*.jpg')  # images from dir
# print(list(files)[:2])
for f in list(files)[:10]:  # first 10 images
    im = cv2.imread(f)
    # im = np.ascontiguousarray(np.asarray(im).transpose((2, 0, 1)))  # HWC to CHW
    # im = torch.tensor(im).float().unsqueeze(0) / 255.0  # to Tensor, to BCWH, rescale
    # im = resize(normalize(im))
    t.append(im)
    classify2(model, size=128, file=im)
# print(t)