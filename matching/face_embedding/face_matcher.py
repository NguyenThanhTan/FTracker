import sys
import numpy as np
import queue as Q
import igraph as ig
import leidenalg as lda
from .matching_utils import *

def inhibit(mark, idx_g, idx_p, probes):
    interval_idxp = get_interval_for_track(probes[idx_p])
    for i, probe in enumerate(probes):
        if i == idx_p:
            continue
        interval_i = get_interval_for_track(probes[i])
        if overlap_temporal(interval_idxp, interval_i):
            mark[i][idx_g] = 1
    return mark

def match_in_batch_hungarian(sess, images_placeholder, phase_train_placeholder, embeddings, name_to_emb, name_to_img, probes, probes_ori, t, threshold = 0.85):
    for probe in probes:
        for frame in probe:
            name_to_emb[frame] = encoding_new(name_to_img[frame], sess, images_placeholder, phase_train_placeholder, embeddings).squeeze()
    cons = []
    for track in probes_ori:
        cons.append(get_interval_for_track(track))

    cost_matrix = np.full((len(probes_ori), len(probes_ori)), 5.0, dtype=np.float32)
    for i in range(len(probes_ori) - 1):
        for j in range(i + 1, len(probes_ori)):
            if not overlap_temporal(cons[i], cons[j]):
                if cons[i][1] < cons[j][1]:
                    cost_matching = compare(name_to_emb, probes[i], probes_ori[i], probes[j], probes_ori[j])
                    cost_matrix[i][j] = cost_matching
                else:
                    cost_matching = compare(name_to_emb, probes[j], probes_ori[j], probes[i], probes_ori[i])
                    cost_matrix[j][i] = cost_matching

    from scipy.optimize import linear_sum_assignment

    cost_matrix[cost_matrix > threshold] = 5

    row_inds, col_inds = linear_sum_assignment(cost_matrix)
    mark = np.zeros((len(probes) * 2,), dtype=np.int32)
    for i, row_ind in enumerate(row_inds):
        if cost_matrix[row_ind][col_inds[i]] < threshold:
            probes.append(probes[row_ind] + probes[col_inds[i]])
            probes_ori.append(probes_ori[row_ind] + probes_ori[col_inds[i]])
            print('match in batch:', row_ind, col_inds[i], cost_matrix[row_ind][col_inds[i]])
            mark[row_ind] = 1
            mark[col_inds[i]] = 1

    res1 = []
    res2 = []
    for i, probe in enumerate(probes):
        if not mark[i]:
            res1.append(probe)
            res2.append(probes_ori[i])
    return res1, res2, get_trackids_from_galleries(res1)


def match_hungarian(sess, images_placeholder, phase_train_placeholder, embeddings, name_to_emb, name_to_img, galleries,
                    reps_g, probes, t, final_res, threshold=0.85):
    probes_ori = probes.copy()
    print('probes: ', t)
    probes = compress_tracks_by_hog(name_to_img, probes)
    probes, probes_ori, t = match_in_batch_hungarian(sess, images_placeholder, phase_train_placeholder, embeddings,
                                                     name_to_emb, name_to_img, probes, probes_ori, t)

    if len(galleries) == 0:
        for i, probe in enumerate(probes_ori):
            final_res = update_trackid(probe, i, final_res)
        return probes, probes_ori, final_res

    cost_matrix = np.full((len(probes), len(galleries)), 5.0, dtype=np.float32)
    for i in range(len(probes)):
        for j in range(len(galleries)):
            cost_matrix[i][j] = compare(name_to_emb, galleries[j], reps_g[j], probes[i], probes_ori[i])

    mark_p = np.zeros((len(probes),))
    from scipy.optimize import linear_sum_assignment

    cost_matrix[cost_matrix > threshold] = 5.0

    row_inds, col_inds = linear_sum_assignment(cost_matrix)
    print('Start Hungarian')
    groups = get_trackids_from_galleries(galleries)
    print('galleries before: ', groups)
    print('probes: ', t)
    for i, row_ind in enumerate(row_inds):
        print('choosen min:', t[row_ind], '->', groups[col_inds[i]], 'with ', cost_matrix[row_ind][col_inds[i]])
        if cost_matrix[row_ind][col_inds[i]] < threshold:
            galleries[col_inds[i]] += probes[row_ind]
            reps_g[col_inds[i]] += probes_ori[row_ind]
            final_res = update_trackid(probes_ori[row_ind], col_inds[i], final_res)
            mark_p[row_ind] = 1
        else:
            final_res = update_trackid(probes_ori[row_ind], len(galleries), final_res)
            galleries.append(probes[row_ind])
            reps_g.append(probes_ori[row_ind])
            mark_p[row_ind] = 1
    groups = get_trackids_from_galleries(galleries)
    print('galleries after Hungarian: ', groups)

    for id, probe in enumerate(probes):
        if not mark_p[id]:
            final_res = update_trackid(probes_ori[id], len(galleries), final_res)
            galleries.append(probes[id])
            reps_g.append(probes_ori[id])
    groups = get_trackids_from_galleries(galleries)
    print('galleries final: ', groups)
    return galleries, reps_g, final_res

def match_in_batch(sess, images_placeholder, phase_train_placeholder, embeddings, name_to_emb, name_to_img, probes, probes_ori, t, threshold = 0.85):
    for probe in probes:
        for frame in probe:
            name_to_emb[frame] = encoding_new(name_to_img[frame], sess, images_placeholder, phase_train_placeholder, embeddings).squeeze()
    cons = []
    for track in probes_ori:
        a = []
        for frame in track:
            a.append(int(frame.split('/')[-1].split('_')[0]))
        a.sort()
        cons.append((a[0], a[-1]))
    pri_q = Q.PriorityQueue()
    for i in range(len(probes_ori) - 1):
        for j in range(i + 1, len(probes_ori)):
            if not overlap_temporal(cons[i], cons[j]):
                if cons[i][1] < cons[j][1]:
                    pri_q.put((compare(name_to_emb, probes[i], probes_ori[i], probes[j], probes_ori[j]), [i, j]))
                else:
                    pri_q.put((compare(name_to_emb, probes[j], probes_ori[j], probes[i], probes_ori[i]), [j, i]))


    mark = np.zeros((len(probes) * 2,), dtype=np.int32)
    while not pri_q.empty():
        min_dist = pri_q.get()
        if min_dist[0] > threshold:
            break
        if mark[min_dist[1][0]] + mark[min_dist[1][1]] > 0:
            continue
        if is_wrong_match(probes, min_dist[1][0], min_dist[1][1]):
            if is_wrong_by_embedding(min_dist[0]):
                print('fault from facenet in batch')
            else:
                print('fault from matching in batch')
        probes.append(probes[min_dist[1][0]] + probes[min_dist[1][1]])
        print('match in batch: ', min_dist)
        probes_ori.append(probes_ori[min_dist[1][0]] + probes_ori[min_dist[1][1]])
        mark[min_dist[1][0]] = 1
        mark[min_dist[1][1]] = 1
    res1 = []
    res2 = []
    for i, probe in enumerate(probes):
        if not mark[i]:
            res1.append(probe)
            res2.append(probes_ori[i])
    return res1, res2, get_trackids_from_galleries(res1)

def match(sess, images_placeholder, phase_train_placeholder, embeddings, name_to_emb, name_to_img, galleries, reps_g,
          probes, t, final_res, thresh=0.85):
    probes_ori = probes.copy()
    probes = compress_tracks_by_hog(name_to_img, probes)
    #probes is compressed
    probes, probes_ori, t = match_in_batch_hungarian(sess, images_placeholder, phase_train_placeholder, embeddings, name_to_emb, name_to_img, probes, probes_ori, t)
    if len(galleries) == 0:
        for i, probe in enumerate(probes_ori):
            final_res = update_trackid(probe, i, final_res)
        return probes, probes_ori, final_res

    m = np.zeros((len(probes), len(galleries)))
    for i in range(len(probes)):
        for j in range(len(galleries)):
            m[i][j] = compare(name_to_emb, galleries[j], reps_g[j], probes[i], probes_ori[i])

    mark_p = np.zeros((len(probes),))
    mark = np.zeros((len(probes), len(galleries)))
    len_gal = len(galleries)

    while (mark_p == 1).all() == False:
        groups = get_trackids_from_galleries(galleries)
        print('galleries before: ', groups)
        print('probes: ', t)
        idx_g = np.argmin(m) % len_gal
        idx_p = int(np.argmin(m) / len_gal)
        while mark[idx_p][idx_g] or mark_p[idx_p] == 1:
            if not m[idx_p][idx_g] == 5 and (mark[idx_p] == 1).all() and not mark_p[idx_p]:
                m[idx_p][idx_g] = 5
                break

            m[idx_p][idx_g] = 5
            idx_g = np.argmin(m) % len_gal
            idx_p = int(np.argmin(m) / len_gal)

        print('choosen min:' ,list(t)[idx_p], '->', groups[idx_g], 'with ', m[idx_p][idx_g])

        if is_wrong_match(None, groups[idx_g], list(t)[idx_p]):
            if is_wrong_by_embedding(m[idx_p][idx_g]):
                print('fault from facenet')
            else:
                print('fault from matching')

        if m[idx_p][idx_g] < thresh:
            galleries[idx_g] += probes[idx_p]
            final_res = update_trackid(probes_ori[idx_p], idx_g, final_res)
            reps_g[idx_g] += probes_ori[idx_p]
            mark = inhibit(mark, idx_g, idx_p, probes_ori)
        else:
            final_res = update_trackid(probes_ori[idx_p], len(galleries), final_res)
            galleries.append(probes[idx_p])
            reps_g.append(probes_ori[idx_p])
        mark_p[idx_p] = 1
        groups = get_trackids_from_galleries(galleries)
        print('galleries after: ', groups)
    return galleries, reps_g, final_res

def match_leiden(name_to_emb, name_to_img, galleries, reps_g, probes, final_res, resolution=1.8):
    probes.sort(key=lambda x: int(x[0].split('/')[-1].split('_')[0]))
    probes_ori = probes.copy()
    probes = compress_tracks_by_hog(name_to_img, probes)

    for probe in probes:
        for frame in probe:
            name_to_emb[frame] = encoding(frame, sess, images_placeholder, phase_train_placeholder, embeddings).squeeze()

    dic = dict()
    for i in range(len(galleries)):
        dic[i] = {frame.split('/')[-2] for frame in galleries[i]}
    for i in range(len(probes)):
        dic[i + len(galleries)] = {frame.split('/')[-2] for frame in probes[i]}
    dist = np.zeros((len(probes), len(galleries)))
    for i in range(len(probes)):
        for j in range(len(galleries)):
            dist[i][j] = compare(name_to_emb, galleries[j], reps_g[j], probes[i], probes_ori[i])
    cons = []
    for track in probes_ori:
        a = []
        for frame in track:
            a.append(int(frame.split('/')[-1].split('_')[0]))
        a.sort()
        cons.append((a[0], a[-1]))
    in_probes = [[] for i in range(len(probes) - 1)]
    for i in range(len(probes)-1):
        for j in range(i+1, len(probes)):
            if not overlap_temporal(cons[i], cons[j]):
                if cons[i][1] < cons[j][1]:
                    in_probes[i].append(compare(name_to_emb, probes[i], probes_ori[i], probes[j], probes_ori[j]))
                else:
                    in_probes[i].append(compare(name_to_emb, probes[j], probes_ori[j], probes[i], probes_ori[i]))
            else:
                in_probes[i].append(3)

    test1 = get_trackids_from_galleries(probes)
    weights = []
    for i in range(len(galleries)):
        weights += (len(galleries)-i-1)*[3] + list(dist[:, i])
    for i in range(len(probes) - 1):
        weights += in_probes[i]
    weights = [3 - x for x in weights]

    g = ig.Graph.Full(len(galleries) + len(probes))
    p = lda.find_partition(g, lda.CPMVertexPartition, weights=weights, resolution_parameter = resolution)

    communities = p.as_cover()[:]
    res1 = [[] for i in communities]
    res2 = [[] for i in communities]
    communities.sort(key=lambda x: x[0])
    res = [{} for i in communities]
    for i, partition in enumerate(communities):
        res[i] = set.union(*[dic[idx] for idx in partition])
        if len(partition) > 1:
            print('probes: ', [dic[partition[i]] for i in range(1, len(partition))], '-> gallery: ', dic[partition[0]])
        for element in partition:
            if element < len(galleries):
                res1[i] += galleries[element]
                res2[i] += reps_g[element]
            else:
                final_res = update_trackid(probe=probes_ori[element - len(galleries)], idt=i, final_res=final_res)
                res1[i] += probes[element - len(galleries)]
                res2[i] += probes_ori[element - len(galleries)]
    return res1, res2, final_res