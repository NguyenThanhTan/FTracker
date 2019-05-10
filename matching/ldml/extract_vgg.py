from face_embedding.face_encoder import FaceEncoder, encoding_new
import os
import numpy as np
import cv2
from face_embedding.compresser import Compresser


working_dir = '/opt/hades/'
face_dir = 'vgg_face2/train/'
face_features_dir = 'vgg_face2/train_features/'

for track in os.listdir(working_dir + face_dir):
    f = []
    scores = []
    for frame in os.listdir(working_dir + face_dir + str(track)):
        score = 1
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


