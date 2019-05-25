import os

#=================#
# utilities
#=================#

EPSILON = 0.000001
def cal_iou(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    # compute intersection
    inter = (xb-xa)*(yb-ya)
    if (xb >= xa) and (yb >= ya):
        a = (box_a[2] - box_a[0])*(box_a[3] - box_a[1])
        b = (box_b[2] - box_b[0])*(box_b[3] - box_b[1])
        iou = inter/(float(a+b-inter) + EPSILON)
        return iou
    else:
        return 0

def to_xyxy(roi):
    x, y, w, h = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
    return (x,y,x+w,y+h)

def to_xywh(box):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    return (xmin,ymin,xmax-xmin,ymax-ymin)

def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
