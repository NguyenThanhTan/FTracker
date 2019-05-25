import os
import cv2
import numpy as np
import pickle
from body_matching.visual_compare import compare_two_tracklets
from body_matching.LOMO.lm import get_img_lomo
from body_matching.ColorNaming.ca import get_img_cn
from body_matching.ColorHist.ch import get_img_hist
from body_matching.Facenet.fn import get_img_facenet

office3 = [[0,4,7,11,15,16,17,18], [1,5,8,9,14,22], [2,3,6,10,12,13,19,20,23,24,25]]
taser1 = [[0,2,3,5], [1,4,6]]
taser2 = [[0,1,3,8,10,14,16], [2,12,13,15,17], [4,7,9], [18], [5,6,11]]
reporters = [[0,1,3,7,14,15,19], [2,10,13,16,20,22], [4,8,9,17], [5,6,11,12,18]]
lights = [[0,2], [1,3,4,5,6,7,8]]
axonvn = [[0], [1,6], [2,4,8], [5], [7]]
drunk = [[0,5,19], [3,4,17,18,20,21], [6,7,9], [8,10,11,12], [14], [13,15,16]]
monkey = [[0,3,4,12,15,21,22], [1,8,16,19], [2,9]]
camden = [[0,8,19,26,40], [1,5,11,16,17,22,25], [2,12,14,23,28,34],
          [4,9,15,18,21,24,27,33], [6],[35,38,39], [20,29,30,31], [32,44]]
wounded = [[0,6,24,34,35,39,45,65,71], [1,5,12,17], [3,13,18,19], [4,23], [7,64],
           [30], [40,42,50,56,61,67]]
shooting = [[1,4,11,13,14,15,17,22,23,24,32,38,39,40], [2,12,18,19,25,26,27,29],
            [3,6,9,10,20,21,30,35,37], [28,33,34]]
fleeing = [[7,15], [4,6,9,12,13,14,16], [5,10,11], [3,8]]
blue = [[0,6], [8,9,14,26], [4,16,17,22,24,27], [15,18,21,23]]
store3 = [[0,4,9], [1,3,5,6,11]]
fleeing2 = [[0,4,15,22,31,59,63,83,85,86,92,99], [1,7,12,18,29,38,67],
            [2,3,19,27,34,43,58,65], [8,11,17,20,26,73,79,84]]
videos_gt = {'axonvn': axonvn,
             'office3': office3,
             'taser1': taser1,
             'taser2': taser2,
             'reporters': reporters,
             'lights': lights,
             'drunk': drunk,
             'monkey': monkey,
             'camden':camden,
             'wounded':wounded,
             'shooting': shooting,
             'fleeing': fleeing,
             'blue': blue,
             'store3': store3,
             'fleeing2': fleeing2}


def get_body_coordinates(f_coors, h, w):
    x1 = int(max(0, f_coors[0] - (f_coors[2] - f_coors[0])/2))
    x2 = int(min(w-1, f_coors[2] + (f_coors[2] - f_coors[0])/2))
    y1 = int(max(0, f_coors[1] + (f_coors[3] - f_coors[1])))
    y2 = int(min(h-1, f_coors[3] + (f_coors[3] - f_coors[1])*2))
    return x1, y1, x2, y2


def create_body_images(working_dir, full_dir, body_dir, video_gt):

    # get shape
    t = os.listdir(working_dir + full_dir + '0/')
    height, width, _ = cv2.imread(working_dir + full_dir + '0/' + t[0]).shape

    for track in sum(video_gt, []):
        t = os.listdir(working_dir + full_dir + str(track))
        for frame in t:
            img = cv2.imread(working_dir + full_dir + str(track) + '/' + frame)
            f_coors = frame.split('_')[1:5]
            f_coors = [int(x) for x in f_coors]
            b_coors = get_body_coordinates(f_coors, height, width)
            if not os.path.exists(working_dir + body_dir + str(track)):
                os.makedirs(working_dir + body_dir + str(track))
            b_name = [str(b_coors[i]) for i in range(len(b_coors))]
            b_name.insert(0, str(frame.split('_')[0]))
            b_name.insert(len(b_name), str(frame.split('_')[5]))
            b_name = '_'.join(b_name)
            cv2.imwrite(working_dir + body_dir + str(track) + '/' + b_name, img[b_coors[1]:b_coors[3], b_coors[0]:b_coors[2]])


def is_detection(path):
    path = path[:-4]  # remove extension
    return float(path.split('_')[-1]) > 0


feature_to_func = {'lomo': get_img_lomo, 'cn': get_img_cn, 'ch': get_img_hist,
                   'facenet': get_img_facenet, 'cn_lm': get_img_cn}


def get_intra_distances(working_dir, body_dir, feature, video_gt):
    distances = []
    for identity in video_gt:
        for i, tl1 in enumerate(identity):
            tl1 = str(tl1)
            if os.path.exists(os.path.join(working_dir, body_dir, feature + tl1 + '.pkl')):
                features1 = load_obj(os.path.join(working_dir, body_dir, feature + tl1))
            else:
                names_tl1 = os.listdir(working_dir + body_dir + tl1 + '/')
                imgs_tl1 = [cv2.imread(working_dir + body_dir + tl1 + '/' + name)
                            for name in names_tl1 if is_detection(name)]
                if len(imgs_tl1) > 20:
                    imgs_tl1 = np.random.choice(imgs_tl1, size=np.clip(len(imgs_tl1), 5, 20), replace=False)
                print(len(imgs_tl1))
                features1 = [feature_to_func[feature](img) for img in imgs_tl1]
                save_obj(features1, os.path.join(working_dir, body_dir, feature + tl1))
            for j, tl2 in enumerate(identity):
                tl2 = str(tl2)
                if j > i:
                    if os.path.exists(os.path.join(working_dir, body_dir, feature + tl2 + '.pkl')):
                        features2 = load_obj(os.path.join(working_dir, body_dir, feature + tl2))
                    else:
                        names_tl2 = os.listdir(working_dir + body_dir + tl2 + '/')
                        imgs_tl2 = [cv2.imread(working_dir + body_dir + tl2 + '/' + name)
                                    for name in names_tl2 if is_detection(name)]
                        if len(imgs_tl2) > 20:
                            imgs_tl2 = np.random.choice(imgs_tl2, size=np.clip(len(imgs_tl2), 5, 20), replace=False)
                        print(len(imgs_tl2))
                        features2 = [feature_to_func[feature](img) for img in imgs_tl2]
                        save_obj(features2, os.path.join(working_dir, body_dir, feature + tl2))
                    distances.append(compare_two_tracklets(features1, features2, feature=feature, tlet1=tl1, tlet2=tl2,
                                                           working_dir=working_dir, body_dir=body_dir))
    return distances


def get_inter_distances(working_dir, body_dir, feature, video_gt):
    distances = []
    for idx1, identity1 in enumerate(video_gt):
        for idx2, identity2 in enumerate(video_gt):
            # if identity1 is not identity2:
            if idx1 < idx2:
                for tl1 in identity1:
                    tl1 = str(tl1)
                    if os.path.exists(os.path.join(working_dir, body_dir, feature + tl1 + '.pkl')):
                        features1 = load_obj(os.path.join(working_dir, body_dir, feature + tl1))
                    else:

                        names_tl1 = os.listdir(working_dir + body_dir + tl1 + '/')
                        imgs_tl1 = [cv2.imread(working_dir + body_dir + tl1 + '/' + name)
                                    for name in names_tl1 if is_detection(name)]
                        if len(imgs_tl1) > 20:
                            imgs_tl1 = np.random.choice(imgs_tl1, size=np.clip(len(imgs_tl1), 5, 20), replace=False)

                        print(len(imgs_tl1))
                        features1 = [feature_to_func[feature](img) for img in imgs_tl1]
                        save_obj(features1, os.path.join(working_dir, body_dir, feature + tl1))
                    for tl2 in identity2:
                        tl2 = str(tl2)
                        if os.path.exists(os.path.join(working_dir, body_dir, feature + tl2 + '.pkl')):
                            features2 = load_obj(os.path.join(working_dir, body_dir, feature + tl2))
                        else:
                            names_tl2 = os.listdir(working_dir + body_dir + tl2 + '/')
                            imgs_tl2 = [cv2.imread(working_dir + body_dir + tl2 + '/' + name)
                                        for name in names_tl2 if is_detection(name)]
                            if len(imgs_tl2) > 20:
                                imgs_tl2 = np.random.choice(imgs_tl2, size=np.clip(len(imgs_tl2), 5, 20), replace=False)
                            print(len(imgs_tl2))
                            features2 = [feature_to_func[feature](img) for img in imgs_tl2]
                            save_obj(features2, os.path.join(working_dir, body_dir, feature + tl2))
                        distances.append(compare_two_tracklets(features1, features2, feature=feature, tlet1=tl1, tlet2=tl2, working_dir = working_dir, body_dir = body_dir))
    return distances


def save_obj(obj, name):
    print('save {}'.format(name))
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    print('load {}'.format(name))
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def main():
    # working_dir = '/Users/ttnguyen/Documents/work/hades/'
    working_dir = '/opt/hades/'
    # run this block when updating get_body_coors function
    # for k, v in videos_gt.items():
    #     body_dir = k + '_body/'
    #     if os.path.exists(working_dir + body_dir):
    #         rmtree(working_dir + body_dir)
    #     full_dir = k + '_full/'
    #     create_body_images(working_dir, full_dir, body_dir, v)

    distances = dict()
    # features = ['facenet', 'cn', 'ch', 'lomo']
    features = ['cn']
    for feature in features:
        print(feature)
        distances[feature] = dict()
        distances[feature]['intra'] = dict()
        distances[feature]['inter'] = dict()
        for k, v in videos_gt.items():
            if feature in ['facenet', 'lomo']:
                body_dir = k + '/'
            else:
                body_dir = k + '_body/'
            distances[feature]['intra'][k] = get_intra_distances(working_dir, body_dir, feature, v)
            distances[feature]['inter'][k] = get_inter_distances(working_dir, body_dir, feature, v)
        save_obj(distances[feature], '{}_m100_dist_video'.format(feature))


main()
