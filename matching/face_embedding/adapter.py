import sys
import numpy as np
import os
from .matching_utils import *
BATCH_SIZE = 32

def batch_tracks_to_probes(batch_tracks, fids):

    set_of_trackid = set()
    for frame in batch_tracks:
        for trackid, geo in frame.items():
            set_of_trackid.add(str(trackid))

    set_of_trackid = list(set_of_trackid)
    probes = [[] for x in range(len(set_of_trackid))]
    trackid_to_idx = dict()
    for i, trackid in enumerate(set_of_trackid):
        trackid_to_idx[trackid] = i

    for i, frame in enumerate(batch_tracks):
        for trackid, geo in frame.items():
            cor = [_ for _ in geo]
            cor[2] += cor[0]
            cor[3] += cor[1]
            path = [str(fids[i])] + [str(_) for _ in cor]
            path = '_'.join(path)
            probes[trackid_to_idx[str(trackid)]].append(str(trackid) + '/' + path)
    probes.sort(key=lambda x: int(get_frameid(x[0])))
    for probe in probes:
        if float(get_score(probe[-1])) < 0 and (int(get_frameid(probe[-1])) != fids[-1] or len(fids) < BATCH_SIZE):
            delete_tail_track(probe, batch_tracks, fids)
    probes = [probe for probe in probes if len(probe)]

    return probes

def detection_tracking_output_to_matching_input(name_to_img, batch_tracks, batch_bitmaps, fids):
    t = set()
    for x in batch_tracks:
        for key,value in x.items():
            t.add(str(key))

    t = list(t)
    probes = [[] for x in range(len(t))]
    trackid_to_idx = dict()

    final_res = {str(fids[i]) : dict() for i in range(len(batch_tracks))}
    for i in range(len(batch_tracks)):
        if len(batch_tracks[i]) != 0:
            for key,value in batch_tracks[i].items():
                final_res[str(fids[i])][str(key)] = {'geometries': [str(value[0]), str(value[1]), str(value[0] + value[2]), str(value[1] + value[3])], 'id': '-1'}

    for i, x in enumerate(t):
        trackid_to_idx[x] = i
    #print('batch bitmaps - {}'.format(batch_bitmaps))

    for i, x in enumerate(batch_tracks):
        for trackid, geo in x.items():
            cor = [_ for _ in geo]
            cor[2] += cor[0]
            cor[3] += cor[1]
            path = [str(fids[i])] + [str(_) for _ in cor]
            path = '_'.join(path)
            name_to_img[str(trackid) + '/' + path] = batch_bitmaps[i][cor[1]:cor[3], cor[0]:cor[2]]
            probes[trackid_to_idx[str(trackid)]].append(str(trackid) + '/' + path)
    probes.sort(key=lambda x: int(get_frameid(x[0])))

    for i,probe in enumerate(probes):
        if len(probe) < 5:
            for frame in probe:
                del final_res[get_frameid(frame)][get_trackid(frame)]
                del batch_tracks[int(get_frameid(frame))%len(batch_tracks)][int(get_trackid(frame))]
            #probes.remove(probe)
    probes = [probe for probe in probes if len(probe) >= 5]

    return probes, t, final_res


def matching_output_to_detection_tracking_output(final_res, batch_tracks):
    res = [dict() for x in range(len(batch_tracks))]

    for i, (key,value) in enumerate(sorted(final_res.items(), key=lambda kv: int(kv[0]))):
        m = dict()
        for k,v in value.items():
            m[k] = v['id']
        if len(batch_tracks[i]) != 0:
            for x, y in batch_tracks[i].items():
                res[i][int(m[str(x)])] = y
    return res

def preprocess_batchtracks(batch_tracks):
    for i, dic in enumerate(batch_tracks):
        dic = {k : v for k,v in dic.items() if v[2] >= 32 and v[3] >= 32}
        batch_tracks[i] = dic
    return batch_tracks
