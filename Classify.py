class Classifier:

    from classifier import *

    b = 'runs/train/exp8/weights/best.pt'
    p = 'GTSRB/Images'

    def classify_sign(path, best):
        l = []
        model = torch.load(best, map_location=torch.device('cpu'))['model'].float()
        files = Path(path).glob('*.jpg')  # images from dir
        for f in list(files)[:10]:  # first 10 images
            tup = classify(model, size=128, file=f)
            classID = tup.argmax().item()
            l.append(classID)
        return l