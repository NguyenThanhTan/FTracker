import numpy as np
import cv2
import os
from body_matching.LOMO.lm import get_img_lomo
from body_matching.ColorNaming.ca import get_img_cn
from body_matching.ColorHist.ch import get_img_hist
from sklearn.metrics.pairwise import cosine_similarity as css
from sklearn.metrics.pairwise import cosine_distances as csd

lomo_dir = '/opt/hades/body_matching/LOMO/'


def cn_lm(pixel_cns):
    import os.path as path
    import json
    import scipy.ndimage
    with open(path.join(lomo_dir, 'config.json'), 'r') as f:
        config = json.load(f)['lomo']
    # print('before', pixel_cns.shape)
    img = np.zeros((config['width'], config['height'], 11))
    scipy.ndimage.interpolation.zoom(input=pixel_cns,
                                     zoom=(config['width']/pixel_cns.shape[0],
                                           config['height']/pixel_cns.shape[1],
                                           1),
                                     output=img,
                                     order=2)
    # print('after', pixel_cns.shape)
    block_size = config['block_size']
    block_step = config['block_step']

    # img = pixel_cns

    row_num = (img.shape[0] - (block_size - block_step)) / block_step
    col_num = (img.shape[1] - (block_size - block_step)) / block_step

    cn_feat = np.array([])
    for row in range(int(row_num)):
        for col in range(int(col_num)):
            img_block = img[
                        row * block_step:row * block_step + block_size,
                        col * block_step:col * block_step + block_size
                        ]
            cn_hist = np.array([])
            # dummy = (0, 128)
            # dummy1 = [_ for _ in range(11)]
            # dummy2 = [dummy[_ % 2] for _ in range(22)]
            # dummy3 = [2 for _ in range(11)]
            # img_block = (img_block * 255).astype(np.uint8)
            # hist = cv2.calcHist([img_block], dummy1, None, dummy3, dummy2)
            #
            # hist = hist.reshape(-1,)
            # cn_hist = np.concatenate([cn_hist, hist], 0)
            # print(cn_hist.shape)
            for i in range(img.shape[2]):
                hist, bins = np.histogram(img_block[:,:,i], bins = 5, range=(0,1))
                hist[0] -= 100
                cn_hist = np.concatenate([cn_hist, hist], 0)

            if col == 0:
                cn_feat_col = cn_hist
            else:
                cn_feat_col = np.maximum(cn_feat_col, cn_hist)

        cn_feat = np.concatenate([cn_feat, cn_feat_col], 0)

    # cn_feat = np.log(cn_feat + 1.0)
    # cn_feat /= np.linalg.norm(cn_feat)
    print(cn_feat)

    return cn_feat

def img_to_cat(pixel_cns):
    # pixel_cns = cn_lm(pixel_cns)
    return np.sum(np.sum(pixel_cns, axis=0), axis=0) / (pixel_cns.shape[0] * pixel_cns.shape[1])


def compare_two_images(img1, img2, **kwargs):
    if kwargs['feature'] == 'lomo':
        f1 = get_img_lomo(img1)
        f2 = get_img_lomo(img2)
    elif kwargs['feature'] == 'ch':
        f1 = get_img_hist(img1)
        f2 = get_img_hist(img2)
    elif kwargs['feature'] == 'cn':
        f1 = get_img_cn(img1)
        f1 = np.sum(np.sum(f1, axis=0), axis=0) / (f1.shape[0] * f1.shape[1])
        f2 = get_img_cn(img2)
        f2 = np.sum(np.sum(f2, axis=0), axis=0) / (f2.shape[0] * f2.shape[1])
    else:
        print('unknown feature')
        return

    res = css(f1.reshape(1, -1), f2.reshape(1, -1))[0]
    print(res)
    return res


def compare_two_features(f1, f2, **kwargs):
    if kwargs['feature'] == 'lomo':
        res = css(f1.reshape(1, -1), f2.reshape(1, -1))[0]
    elif kwargs['feature'] == 'ch':
        res = cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL)
    elif kwargs['feature'] == 'cn':
        f1 = np.sum(np.sum(f1, axis=0), axis=0) / (f1.shape[0] * f1.shape[1])
        f2 = np.sum(np.sum(f2, axis=0), axis=0) / (f2.shape[0] * f2.shape[1])
        res = css(f1.reshape(1, -1), f2.reshape(1, -1))[0]
    elif kwargs['feature'] == 'facenet':
        res = css(f1.reshape(1, -1), f2.reshape(1, -1))[0]
    else:
        print('unknown feature')
        return

    return res
def load_obj(name):
    print('load {}'.format(name))
    import pickle
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_obj(obj, name):
    print('save {}'.format(name))
    import pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def compare_two_tracklets(tl1, tl2, **kwargs):

    if kwargs['feature'] == 'lomo':
        distances = list(css(tl1, tl2).reshape(-1,))
    elif kwargs['feature'] == 'ch':
        distances = []
        for f1 in tl1:
            for f2 in tl2:
                distances.append(cv2.compareHist(f1, f2, cv2.HISTCMP_CORREL))
    elif kwargs['feature'] == 'cn':
        tl1 = [get_img_cn(f1) for f1 in tl1]
        tl2 = [get_img_cn(f2) for f2 in tl2]
        distances = list(css(tl1, tl2).reshape(-1,))
    elif kwargs['feature'] == 'cn_lm':
        distances = list(css(tl1, tl2).reshape(-1, ))
    elif kwargs['feature'] == 'facenet':
        distances = list(css(tl1, tl2).reshape(-1, ))
    else:
        print('unknown feature')
        return

    return distances
