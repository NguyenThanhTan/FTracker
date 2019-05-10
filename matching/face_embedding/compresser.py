import sys
import numpy as np
import os
import cv2
from scipy import misc
from settings import logger
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, MeanShift
from skimage.feature import local_binary_pattern
from .matching_utils import *



class Compresser():

    feature_to_threshold = {'hog': 5.0, 'lbp': 170.0}
    @classmethod
    def get_nof_clusters(cls, track):
        return int(track.shape[0]/10 + 1)

    @classmethod
    def compress_clustering(cls, features, scores, type='average'):
        if features.shape[0] < 2:
            return [0]
        nof_clusters = cls.get_nof_clusters(features)
        features = features.reshape((features.shape[0], features.shape[1]))
        scores = scores.reshape((scores.shape[0],))
        clus = AgglomerativeClustering(n_clusters=nof_clusters, linkage=type)
        clus.fit(features)
        ids_choosen = []
        for j in range(nof_clusters):
            ids_score_j = np.where(clus.labels_ == j)[0]
            scores_j = scores[ids_score_j]
            tracklet_j = features[ids_score_j]
            assert len(scores_j) == len(tracklet_j)
            nof_rep = min(2, len(scores_j))
            ids_choosen.extend(ids_score_j[np.argsort(scores_j)[-nof_rep:]])
        ids_choosen = sorted(ids_choosen)
        logger.debug('from: {} to : {} with indices: {}'.format(len(features), len(ids_choosen), ids_choosen))
        return ids_choosen
    @classmethod
    def compress_sequential(cls , features, type='hog'):
        THRESHOLD = cls.feature_to_threshold[type]
        ids_choosen = [0]
        for i in range(len(features) - 1):
            if np.linalg.norm(features[i+1] - features[ids_choosen[-1]]) > THRESHOLD:
                ids_choosen.append(i+1)

        if (len(features) - 1) not in ids_choosen:
            ids_choosen.append(len(features) - 1)
        logger.debug('from: {} to : {} with indices: {}'.format(len(features), len(ids_choosen), ids_choosen))
        return ids_choosen
    @classmethod
    def compute_hog_features(cls, list_of_images):
        cell_size = (8, 8)  # h x w in pixels
        block_size = (2, 2)  # h x w in cells
        nbins = 9  # number of orientation bins
        hog = cv2.HOGDescriptor(_winSize=(80,80),#(img.shape[1] // cell_size[1] * cell_size[1], img.shape[0] // cell_size[0] * cell_size[0]),
                                        _blockSize=(block_size[1] * cell_size[1],
                                                    block_size[0] * cell_size[0]),
                                        _blockStride=(cell_size[1], cell_size[0]),
                                        _cellSize=(cell_size[1], cell_size[0]),
                                        _nbins=nbins)
        hog_features = []
        for img in list_of_images:
            # logger.debug('Image shape: {}'.format(img.shape))
            img = cv2.resize(img, (80, 80))
            h = hog.compute(img)
            hog_features.append(h)
        return hog_features
    @classmethod
    def feature_matrix_to_histogram_vector(cls, lbp, x, y):
        a = np.array([])
        rx = int(lbp.shape[0] / x)
        ry = int(lbp.shape[1] / y)
        lbp = lbp.astype(np.float32)
        for i in range(x):
            for j in range(y):
                hist1 = cv2.calcHist([lbp[i * rx:(i + 1) * rx, j * ry:(j + 1) * ry]], [0], None, [256], [0, 256])
                hist1 = hist1.reshape((hist1.shape[0],))
                a = np.concatenate((a, hist1))
        return a
    @classmethod
    def compute_lbp_features(cls, list_of_images):
        lbp_features = []
        for img in list_of_images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (80, 80))
            lbp = local_binary_pattern(img, 8, 2)
            lbph = feature_matrix_to_histogram_vector(lbp, 4, 4)
            lbp_features.append(lbph)

        return lbp_features