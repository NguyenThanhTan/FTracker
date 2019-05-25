import sys
sys.path.append('/opt/hades/face_embedding/')
import numpy as np
from .matching_utils import *


def compare_iou(gallery, probe):
    before = gallery[-3:]
    after = probe[:3]
    if int(before[-1].split('/')[-1].split('_')[0]) + 10 < int(after[0].split('/')[-1].split('_')[0]):
        return 0
    bb1 = np.zeros((4,))
    bb2 = np.zeros((4,))
    for i in before:
        for j in range(1,5):
            bb1[j-1] += int(i.split('/')[-1].split('_')[j])
    for i in after:
        for j in range(1,5):
            bb2[j-1] += int(i.split('/')[-1].split('_')[j])
    bb1 /= len(before)
    bb2 /= len(after)
    ixmin = np.maximum(bb1[0], bb2[0])
    iymin = np.maximum(bb1[1], bb2[1])
    ixmax = np.minimum(bb1[2], bb2[2])
    iymax = np.minimum(bb1[3], bb2[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inter = iw * ih
    uni = ((bb2[2] - bb2[0] + 1.) * (bb2[3] - bb2[1] + 1.) + (bb1[2] - bb1[0] + 1.) * (bb1[3] - bb1[1] + 1.) - inter)
    over = inter / uni
    return over

def compare_trackid(gallery, probe):
    trackid_g = {int(frame.split('/')[-2]) for frame in gallery}
    dic = dict()
    for track in trackid_g:
        dic[track] = 0
    for frame in gallery:
        dic[int(frame.split('/')[-2])] += 1
    trackid_g = {k for k,v in dic.items() if v > 10}
    trackid_pro = {int(frame.split('/')[-2]) for frame in probe}
    inters = set.intersection(trackid_g, trackid_pro)
    return len(inters)

def compare_embedding(name_to_emb, gallery, probe):
    g = [name_to_emb[x] for x in gallery]
    g = np.array(g)
    p = [name_to_emb[x] for x in probe]
    p = np.array(p)
    return average_compare(g, p)

def compare_motion(reps_g, reps_p):
    first_diag = [get_diag(get_coordinates(x)) for x in reps_g]
    second_diag = [get_diag(get_coordinates(x)) for x in reps_p]
    norm_value = np.mean(first_diag[-2:] + second_diag[:2])
    norm_value *= 2

    first_inde = [int(get_frameid(x)) for x in reps_g]
    second_inde = [int(get_frameid(x)) for x in reps_p]

    first_dex = [get_center(get_coordinates(x))[0] for x in reps_g]
    second_dex = [get_center(get_coordinates(x))[0] for x in reps_p]
    first_dey = [get_center(get_coordinates(x))[1] for x in reps_g]
    second_dey = [get_center(get_coordinates(x))[1] for x in reps_p]

    forward_x = interp1d(first_inde[-4:], first_dex[-4:], kind='linear', fill_value='extrapolate')
    forward_y = interp1d(first_inde[-4:], first_dey[-4:], kind='linear', fill_value='extrapolate')
    backward_x = interp1d(second_inde[:4], second_dex[:4], kind='linear', fill_value='extrapolate')
    backward_y = interp1d(second_inde[:4], second_dey[:4], kind='linear', fill_value='extrapolate')

    middle = int((second_inde[0] - first_inde[-1]) / 2) + first_inde[-1] + 1
    range_p = np.arange(middle - 2, middle + 2)
    pred_fx = forward_x(range_p)
    pred_bx = backward_x(range_p)
    pred_fy = forward_y(range_p)
    pred_by = backward_y(range_p)

    dist_x = np.linalg.norm(pred_fx - pred_bx) / norm_value
    dist_y = np.linalg.norm(pred_fy - pred_by) / norm_value

    total_dist = math.log(dist_x + dist_y)
    #if total_dist < -1.0:
    #    return -total_dist
    return total_dist

def compare(name_to_emb, gallery, rep_g, probe, rep_b, weight_iou = 0.2, weight_track = 0.2, weight_motion = 0.2):

    if len(rep_g) > 4 and len(rep_b) > 4 and int(get_frameid(rep_g[-1])) + 10 > int(get_frameid(rep_b[0])):
        motion_affinity = compare_motion(rep_g, rep_b)
    else:
        motion_affinity = 0
    appearance_affinity = compare_embedding(name_to_emb, gallery, probe)
    print('appearance affinity between {} and {}: '.format(get_trackids_from_galleries([rep_g]),
                                                           get_trackids_from_galleries([rep_b])),
                                                            appearance_affinity)
    print('motion affinity between {} and {}: '.format(get_trackids_from_galleries([rep_g]),
                                                       get_trackids_from_galleries([rep_b])),
                                                            motion_affinity)
    if motion_affinity > -1:
        motion_affinity = 0
    return  appearance_affinity + weight_motion*motion_affinity - weight_track*compare_trackid(rep_g, rep_b)

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
    #print(v, v1, v2)
    assert v - (v1 + v2) > 0, (v, v1, v2)
    return (v - (v1 + v2))/v