import os.path as path
import json
from body_matching.LOMO import lomo
import cv2


with open(path.join(path.dirname(__file__), 'config.json'), 'r') as f:
    config = json.load(f)


def get_img_lomo(img):

    img = cv2.resize(img, (config['lomo']['height'], config['lomo']['width']))
    lomo_desc = lomo.LOMO(img, config)

    return lomo_desc