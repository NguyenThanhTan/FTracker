import numpy as np
from app.settings import logger

def get_center(abcd):
    return ((int(abcd[2]) - int(abcd[0])) / 2 + int(abcd[0]), (int(abcd[3]) - int(abcd[1])) / 2 + int(abcd[1]))

def get_score(a):
    return float(a.split('/')[-1].split('_')[5])

def get_frameid(a):
    return a.split('/')[-1].split('_')[0]
def get_trackid(a):
    return a.split('/')[-2]

def get_coordinates(a):
    return a.split('/')[-1].split('_')[1:5]


def get_diag(coordinates):
    return np.sqrt(
        np.square(int(coordinates[2]) - int(coordinates[0])) + np.square(int(coordinates[3]) - int(coordinates[1])))


def overlap_temporal(a, b):
    if a[1] > b[1]:
        a, b = b,a
    if b[0] <= a[1]:
        return 1
    return 0

def get_interval_for_track(track):
    interval = []
    for frame in track:
        interval.append(int(get_frameid(frame)))
    interval.sort()
    return interval[0], interval[-1]

def no_compress(list_of_tracks, list_of_scores=None):
    return list_of_tracks

def get_nof_clusters(track):
    return int(track.shape[0]/20 + 1)

def ward_compress(list_of_tracks, list_of_scores):
    res = []
    for i, t in enumerate(list_of_tracks):
        nof_clusters = get_nof_clusters(t)
        #logger.debug('clusters: ', nof_clusters)
        #logger.debug('track_length: ', len(t))
        clus = AgglomerativeClustering(n_clusters=nof_clusters, linkage='ward')
        clus.fit(t)
        l = []
        for j in range(nof_clusters):
            scores = list_of_scores[i][np.where(clus.labels_ == j)[0]].copy()
            track = list_of_tracks[i][np.where(clus.labels_ == j)[0]].copy()
            assert len(scores) == len(track), (len(scores), len(track))
            nof_rep = min(5, len(scores))
            track = track[np.argsort(scores)[-nof_rep:]]
            l.append(track)
        res.append(np.vstack(l))
        #logger.debug(res[i].shape)

    return res
def average_compress(list_of_tracks, list_of_scores):
    res = []
    for i, t in enumerate(list_of_tracks):
        nof_clusters = get_nof_clusters(t)
        #logger.debug('clusters: ', nof_clusters)
        #logger.debug('track_length: ', len(t))
        if len(t) == 0:
            res.append(np.array([]))
            continue
        clus = AgglomerativeClustering(n_clusters=nof_clusters, linkage='average')
        clus.fit(t)
        l = []
        for j in range(nof_clusters):
            scores = list_of_scores[i][np.where(clus.labels_ == j)[0]].copy()
            track = list_of_tracks[i][np.where(clus.labels_ == j)[0]].copy()
            assert len(scores) == len(track), (len(scores), len(track))
            nof_rep = min(5, len(scores))
            track = track[np.argsort(scores)[-nof_rep:]]
            l.append(track)
        res.append(np.vstack(l))
        #logger.debug(res[i].shape)

    return res

def average_compare(t1, t2):
    pairs = t1.shape[0] * t2.shape[0]
    if pairs == 0:
        return 2
    dist = 0.0
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            dist += np.linalg.norm(t1[i] - t2[j])
    dist /= pairs
    return dist
def min_compare(t1, t2):
    dist = []
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            dist.append(np.linalg.norm(t1[i] - t2[j]))
    dist.sort()
    res = 0.0
    avg_num = min(5, len(dist))
    for i in range(avg_num):
        res += dist[i]
    return res/avg_num
def max_compare(t1, t2):
    dist = []
    for i in range(t1.shape[0]):
        for j in range(t2.shape[0]):
            dist.append(np.linalg.norm(t1[i] - t2[j]))
    dist.sort(reverse=True)
    res = 0.0
    avg_num = min(5, len(dist))
    for i in range(avg_num):
        res += dist[i]
    return res/avg_num
def variance_compare(t1, t2):
    v1 = 0.0
    m1 = np.average(t1, axis=0)
    v2 = 0.0
    m2 = np.average(t2, axis=0)
    for i in range(t1.shape[0]):
        v1 += np.linalg.norm(t1[i] - m1)
    #v1 /= t1.shape[0]
    for i in range(t2.shape[0]):
        v2 += np.linalg.norm(t2[i] - m2)
    #v2 /= t2.shape[0]

    new = np.vstack((t1, t2))
    n_m = np.average(new, axis=0)
    v = 0.0
    for i in range(new.shape[0]):
        v += np.linalg.norm(new[i] - n_m)
    #v /= new.shape[0]
    #logger.debug(v, v1, v2)
    assert v - (v1 + v2) > 0, (v, v1, v2)
    return (v - (v1 + v2))/v
def feature_matrix_to_histogram_vector(lbp, x, y):
    a = np.array([])
    rx = int(lbp.shape[0]/x)
    ry = int(lbp.shape[1]/y)
    lbp = lbp.astype(np.float32)
    for i in range(x):
        for j in range(y):
            #hist, bins = np.histogram(lbp[i*rx:(i+1)*rx, j*ry:(j+1)*ry].ravel(), bins=range(0,256))
            #logger.debug('hist', hist.shape)
            hist1 = cv2.calcHist([lbp[i*rx:(i+1)*rx, j*ry:(j+1)*ry]], [0], None, [256], [0,256])
            #logger.debug('hist1', hist1.shape)
            hist1 = hist1.reshape((hist1.shape[0],))
            a = np.concatenate((a, hist1))
    return a# / np.linalg.norm(a)

def compress_by_lbp_feature(names, lbphs):
    id_choosen = [0]
    timer = Timer()
    for i in range(len(lbphs) - 1):
        timer.tic()
        if np.linalg.norm(lbphs[i+1] - lbphs[id_choosen[-1]]) > 200:
            id_choosen.append(i+1)
        timer.toc()
    res = []
    #logger.debug(timer.average_time)
    for i, x in enumerate(names):
        if i in id_choosen:
            res.append(x)
    if (len(names) - 1) not in id_choosen:
        id_choosen.append(len(names) - 1)
    #logger.debug(id_choosen)
    #logger.debug('from: ', len(names), " to: ", len(id_choosen))
    return res
def compress_by_hog_feature(names, hs):
    id_choosen = [0]
    for i in range(len(hs) - 1):
        if np.linalg.norm(hs[i+1] - hs[id_choosen[-1]]) > 5.0:
            id_choosen.append(i+1)
    if (len(names) - 1) not in id_choosen:
        id_choosen.append(len(names) - 1)
    res = []
    for i, x in enumerate(names):
        if i in id_choosen:
            res.append(x)
    logger.debug('from: ', len(names), " to: ", len(id_choosen), "with indices: ", id_choosen)
    return res

### =========================== ###
### =========================== ###

def is_tracks_same_id(i, j, g):
    for l in g:
        if i in l and j in l:
            return 1
    return 0

def prediction_matrix_from_prediction_list(g):
    num_of_tracks = 0
    for x in g:
        num_of_tracks = max(num_of_tracks, max(x))
    mat = np.zeros((num_of_tracks,num_of_tracks))
    for i in range(num_of_tracks):
        for j in range(num_of_tracks):
            mat[i][j] = is_tracks_same_id(i, j, g)
    return mat

def get_trackids_from_galleries(galleries):
    res = []
    for gal in galleries:
        a = {int(x.split('/')[-2]) for x in gal}
        res.append(list(a))
    return res
def is_wrong_match(probes, idx1, idx2):
    if probes != None:
        id1 = list({int(x.split('/')[-2]) for x in probes[idx1]})
        id2 = list({int(x.split('/')[-2]) for x in probes[idx2]})
    else:
        id1 = idx1
        id2 = idx2
    g = [[i for i in range(100)]]
    for i in id1:
        for j in id2:
            if not is_tracks_same_id(i, j, g):
                return 1
    return 0
def is_wrong_by_embedding(distance, mean = 0.85, std = 0.05):
    return distance > mean + 2 * std
def update_trackid(probe, idt, final_res):
    for x in probe:
        frame_id = get_frameid(x)
        track_id = get_trackid(x)
        final_res[frame_id][track_id]['id'] = str(idt)
    return final_res
def delete_tail_track(probe, batch_tracks, fids, refine=False, id=None):
    if not refine:
        id = int(get_trackid(probe[0]))
    else:
        assert id is not None
        id = str(id)
    while(len(probe) > 0 and float(get_score(probe[-1])) < 0):
        frameid = int(get_frameid(probe[-1])) - fids[0]
        logger.debug('refine trackid {} in frame {}'.format(id, int(get_frameid(probe[-1]))))
        del batch_tracks[frameid][id]
        del probe[-1]
