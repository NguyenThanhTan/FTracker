from face_embedding.face_encoder import FaceEncoder, encoding_new
import os
import numpy as np
import cv2
from face_embedding.compresser import Compresser
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
blue = [[0,6], [1,2,5,8,9,14,26], [4,16,17,22,24,27], [15,18,21,23]]
store3 = [[0,4,9], [1,3,5,6,11]]
fleeing2 = [[0,4,15,22,31,59,63,83,85,86,92,99], [1,7,12,18,29,38,67],
            [2,3,19,27,34,43,58,65], [8,11,17,20,26,73,79,84]]
working_dir = '/opt/hades/'
face_dir = 'fleeing2/'
face_features_dir = 'fleeing2_features/'

def get_detection_score(x):
    return float(x[:-4].split('/')[-1].split('_')[5])

for track in sum(fleeing2, []):
    f = []
    scores = []
    for frame in os.listdir(working_dir + face_dir + str(track)):
        score = get_detection_score(frame)
        if score < 0:
            continue
        img = cv2.imread(working_dir + face_dir + str(track) + '/' + frame)
        scores.append(score)
        f.append(encoding_new(img, *(FaceEncoder.encoder)).squeeze())
    f = np.array(f)
    scores = np.array(scores)
    assert len(f) == len(scores)
    ids = Compresser.compress_clustering(f, scores)
    f = f[ids]
    if not os.path.exists(working_dir + face_features_dir):
        os.makedirs(working_dir + face_features_dir)
    np.save(file=working_dir + face_features_dir + str(track), arr=f)


