import cv2
import torch
import json
import time
from RSD_Classification import *

YOLO_PATH = r"C:\Users\kakik\Desktop\RSD-yolov5"
JSON_PATH = 'GTSRB/className.json'
# Model
model = torch.hub.load(YOLO_PATH, 'custom', path='runs/train/ev/yolov5s_results_g2/weights/best.pt', source='local')
model.conf = 0.4

# Image
img = 'eval/image2.jpg'
v = 'eval/Vid_trim.mp4'


# Opening JSON file
with open(JSON_PATH) as json_file:
    data = json.load(json_file)

# Vid
def on_video(vid):
    cap = cv2.VideoCapture(vid)

    # FPS
    prev_f = 0
    new_f = 0

    # f_width = int(cap.get(3))
    # f_height = int(cap.get(4))
    # f_size = (f_width, f_height)

    # save_v = cv2.VideoWriter('inference.mp4', cv2.VideoWriter_fourcc(*"MJPG"), 10, f_size)

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, image = cap.read()
        imager = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # b2 = inference(imager)
        # print(b2)
        box = inference(imager).values.tolist()
        bbox = process_img(imager, box)
        for i, _ in enumerate(box):
            x1 = int(bbox[i][0])
            y1 = int(bbox[i][1])
            x2 = int(bbox[i][2])
            y2 = int(bbox[i][3])
            cf = float(bbox[i][4])
            cl = int(bbox[i][7])
            draw_box(image, x1, y1, x2, y2, cf, cl)
        frame = image_resize(image, height=1080)
        cv2.imshow("inf", image)

        # calculate FPS
        new_f = time.time()
        fps = 1 / (new_f - prev_f)
        prev_f = new_f
        print(fps)
        cv2.putText(img=image, text=f"{fps}", org=(500, 500), color=(255,0,0),
                    fontScale=0.5, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        # save_v.write(image)
        if ret:
            # cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        # Break the loop
        else:
            break
    cap.release()
    # save_v.release()
    cv2.destroyAllWindows()


def on_image(im):
    image = cv2.imread(im)
    imager = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    b2 = inference(imager)
    print(b2)
    box = inference(imager).values.tolist()
    bbox = process_img(image, box)
    for i, _ in enumerate(box):
        x1 = int(bbox[i][0])
        y1 = int(bbox[i][1])
        x2 = int(bbox[i][2])
        y2 = int(bbox[i][3])
        cf = float(bbox[i][4])
        cl = int(bbox[i][5])
        draw_box(image, x1, y1, x2, y2, cf, cl)
    cv2.imshow("inf", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def inference(im):
    results = model(im, size=640)
    # results.show()
    # results.print()
    crops = results.crop(save=False)
    #print(f'cc{crops}')
    #print(crops['cls'])
    # for c in crops:
    #     print(c['im'])
    return results.pandas().xyxy[0]


def draw_box(im, x1, y1, x2, y2, cf, cl):
    if cl == 0:
        RGB = (0, 0, 255)
        cv2.putText(img=im, text=f"{cf:.2f}...car", org=(x1, y1 - 10), color=RGB,
                    fontScale=0.5, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    else:
        RGB = (0, 255, 0)
        cv2.putText(img=im, text=f"{cf:.2f}...{get_name(cl)}", org=(x1, y1 - 10), color=RGB,
                    fontScale=0.5, thickness=1, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    cv2.rectangle(im, (x1, y1), (x2, y2), RGB, 1)


def get_name(class_id):
    return data[str(class_id)]


# on_image(img)
on_video(v)
