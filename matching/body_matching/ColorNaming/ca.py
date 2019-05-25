import scipy.io
import numpy as np
import os.path as path
matrix = scipy.io.loadmat(path.join(path.dirname(__file__), "w2c.mat"))['w2c']

assert matrix.shape[0] == 32768

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


def get_pixel_cn(pixel):
    """Return color name for a pixel (default BGR as in OpenCV)."""

    BLUE = 0
    GREEN = 1
    RED = 2

    k = (pixel[RED] // 8) + 32 * (pixel[GREEN] // 8) + 32 * 32 * (pixel[BLUE] // 8)
    return matrix[k]


def get_img_cn(img):
    cn = np.zeros((img.shape[0], img.shape[1], 11))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            cn[i][j] = get_pixel_cn(img[i][j])
    return cn