import time
from .matching_utils import *
from .compresser import Compresser
from .face_encoder import FaceEncoder, encoding_new
from .adapter import batch_tracks_to_probes

from scipy.interpolate import interp1d
import math


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def probe_to_tracklet(probe, batch_bitmaps, batch_tracks, fids):
    tracklet = Tracklet(batch_bitmaps, fids, batch_tracks, probe)
    return tracklet


def main(matcher, batch_tracks, batch_bitmaps, fids):
    bttp = Timer()
    hung = Timer()
    bttp.tic()
    probes = batch_tracks_to_probes(batch_tracks, fids)
    tracklets = [probe_to_tracklet(probe, batch_bitmaps, batch_tracks, fids) for probe in probes]
    bttp.toc()
    print('bttp time', bttp.average_time) # small compared to hung time
    hung.tic()
    matcher.new_match_hungarian(tracklets)
    hung.toc()
    print('hung time', hung.average_time)
    return batch_tracks


class TrackletState:
    Tentative = 1  # for next batch
    Confirmed = 2  # valid to have id
    Discarded = 3  # invalid to have id
    Assigned = 4  # has been assigned id


class Tracklet():
    """This class represents a track of faces in consecutive frames."""

    def get_detection_indices(self, threshold=0):
        indices = [int(get_frameid(frame)) for frame in self.probe if get_score(frame) > threshold]
        return indices

    def get_detection_boxes(self, threshold=0):
        boxes = [get_coordinates(frame) for frame in self.probe if get_score(frame) > threshold]
        return boxes

    def get_start_detection(self, threshold=0):
        for frame in self.probe:
            if get_score(frame) > threshold:
                return int(get_frameid(frame))
        return -1

    def get_stop_detection(self, threshold=0):
        for frame in reversed(self.probe):
            if get_score(frame) > threshold:
                return int(get_frameid(frame))
        return -1

    def get_stop_track(self):
        return int(get_frameid(self.probe[-1]))

    def get_nof_detections(self, threshold=0):
        nof_detections = 0
        for frame in self.probe:
            if get_score(frame) > threshold:
                nof_detections += 1
        return nof_detections

    def get_detection_scores(self, threshold=0):
        scores = [get_score(frame) for frame in self.probe if get_score(frame) > threshold]
        return scores

    def get_trackid(self):
        return int(get_trackid(self.probe[0]))

    def get_patches(self):
        patches = []
        for frameid, box in zip(self.get_detection_indices(), self.get_detection_boxes()):
            box = [int(cor) for cor in box]
            img = self.batch_bitmaps[frameid - self.fids[0]][box[1]:box[3], box[0]:box[2]]
            # logger.debug(box)
            # logger.debug(self.batch_bitmaps[frameid - self.fids[0]].shape)
            # logger.debug(img.shape)
            patches.append(img)
        return patches

    def delete_tail_track(self):
        # logger.debug(self.probe)
        # for _ in self.batch_tracks:
        #     logger.debug(_)
        while get_score(self.probe[-1]) < 0:
            frameid = int(get_frameid(self.probe[-1])) - self.fids[0]
            if str(self.id) in self.batch_tracks[frameid] and \
                    self.batch_tracks[frameid][str(self.id)][4] < 0:
                logger.debug(
                    "early stop in delete_tail_track for track {} with id {}".format(self.get_trackid(), self.id))
                del self.batch_tracks[frameid][str(self.id)]
                del self.probe[-1]
            else:
                break

    def __init__(self, batch_bitmaps, fids, batch_tracks, probe, state=TrackletState.Tentative):

        self.fids = fids
        self.probe = probe
        self.batch_tracks = batch_tracks
        self.batch_bitmaps = batch_bitmaps
        self.id = None

        logger.debug('init trackid {} in {} -> {}'.format(self.get_trackid(), fids[0], fids[-1]))
        logger.debug('from {} to {}: '.format(self.get_start_detection(), self.get_stop_detection()))
        logger.debug('with ids: {}'.format(self.get_detection_indices()))
        logger.debug('number of detections {} per track {}'.format(self.get_nof_detections(), len(self.probe)))

        self.state = state

        self.ligth_features_list = []
        self.lbp_features = []
        self.features_list = []  # extracted features for faces in the tracklet

    def merge_tracklet(self, tracklet):
        assert self.get_stop_detection() < tracklet.get_start_detection()
        assert self.get_trackid() == tracklet.get_trackid()

        logger.debug('before: {}'.format(len(self.get_features_list())))
        self.features_list = self.get_features_list() + tracklet.get_features_list()
        logger.debug('after: {}'.format(len(self.get_features_list())))

        self.fids += tracklet.fids
        self.probe += tracklet.probe
        self.batch_tracks += tracklet.batch_tracks
        self.batch_bitmaps += tracklet.batch_bitmaps
        self.ligth_features_list += tracklet.ligth_features_list

    def set_id(self, label):
        self.id = label
        for frame_info in self.batch_tracks:
            if self.get_trackid() in frame_info:
                if str(self.id) in frame_info:
                    assert not (frame_info[str(self.id)][4] > 0 and frame_info[self.get_trackid()][4] > 0), \
                        "overlapped tracklets setting id, id {} and {}".format(str(self.id), frame_info)
                    if frame_info[str(self.id)][4] <= 0:
                        frame_info[str(self.id)] = frame_info.pop(self.get_trackid())
                    else:
                        frame_info.pop(self.get_trackid())
                        continue
                else:
                    frame_info[str(self.id)] = frame_info.pop(self.get_trackid())

    def overlaps_with(self, other):
        """Check if this tracklet overlaps with another tracklet."""

        a, b = self.get_start_detection(), self.get_stop_detection()
        c, d = other.get_start_detection(), other.get_stop_detection()
        if c <= a <= d: return True
        if a <= c <= b: return True
        return False

    def compress_by_hog_feature(self, features):
        ids_choosen = [0]
        for i in range(len(features) - 1):
            if np.linalg.norm(features[i + 1] - features[ids_choosen[-1]]) > 4.0:
                ids_choosen.append(i + 1)

        if (len(features) - 1) not in ids_choosen:
            ids_choosen.append(len(features) - 1)
        logger.debug('from: ', len(features), " to: ", len(ids_choosen), "with indices: ", ids_choosen)
        return ids_choosen

    def motion(self, other):
        first_diag = [get_diag(box) for box in self.get_detection_boxes()]
        second_diag = [get_diag(box) for box in other.get_detection_boxes()]

        norm_value = np.mean(first_diag[-2:] + second_diag[:2])
        norm_value = min(200, norm_value)
        norm_value *= 2

        first_inde = self.get_detection_indices()
        second_inde = other.get_detection_indices()

        first_dex = [get_center(box)[0] for box in self.get_detection_boxes()]
        second_dex = [get_center(box)[0] for box in other.get_detection_boxes()]
        first_dey = [get_center(box)[1] for box in self.get_detection_boxes()]
        second_dey = [get_center(box)[1] for box in other.get_detection_boxes()]

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

        # dist_x = np.linalg.norm(pred_fx - pred_bx) / norm_value
        # dist_y = np.linalg.norm(pred_fy - pred_by) / norm_value
        sum = 0
        for pair in zip(pred_fx - pred_bx, pred_fy - pred_by):
            sum += np.sqrt(np.square(pair[0]) + np.square(pair[1]))
        sum /= norm_value * 4

        # total_dist = math.log(dist_x + dist_y)
        total_dist = math.log(sum + 1e-5)
        return total_dist

    def compare_motion(self, other):

        first_diag = [get_diag(box) for box in self.get_detection_boxes()]
        second_diag = [get_diag(box) for box in other.get_detection_boxes()]

        if np.abs(np.mean(first_diag[-2:]) - np.mean(second_diag[:2])) > 100:
            logger.debug('big boxes {} and small boxes{}'.format(self.get_trackid(), other.get_trackid()))
            return 5

        if self.get_nof_detections() < 2 or other.get_nof_detections() < 2 or min(
                np.abs(self.get_start_detection() - other.get_stop_detection()),
                np.abs(self.get_stop_detection() - other.get_start_detection())) > 10:
            return 0
        if self.get_start_detection() < other.get_start_detection():
            return self.motion(other)
        return other.motion(self)

    def compare_appearance(self, other):
        if self.get_start_detection() < other.get_start_detection():
            features_self = self.get_features_list()[-5:]
            features_other = other.get_features_list()[:5]
        else:
            features_self = self.get_features_list()[:5]
            features_other = self.get_features_list()[-5:]
        return average_compare(np.array(features_self), np.array(features_other))

    def compare(self, other):
        track_affinity = 0
        if self.get_trackid() == other.get_trackid():
            track_affinity = 1

        motion_affinity = self.compare_motion(other)
        appearance_affinity = self.compare_appearance(other)
        logger.debug(
            'motion affinity between {} and {} is: {}'.format(self.get_trackid(), other.get_trackid(), motion_affinity))
        logger.debug('appearance affinity between {} and {} is: {}'.format(self.get_trackid(), other.get_trackid(),
                                                                           appearance_affinity))
        # return not track_affinity
        return appearance_affinity + motion_affinity * 0.2 - track_affinity * 0.2

    def compute_features_list(self, type='hog'):
        logger.debug("Compress features list for trackid {}".format(self.get_trackid()))
        self.ligth_features_list = Compresser.compute_hog_features(self.get_patches())
        features = self.ligth_features_list
        scores = self.get_detection_scores()
        logger.debug('compress tracklet {}'.format(self.get_trackid()))
        ids = Compresser.compress_clustering(np.array(features), np.array(scores))
        # ids = Compresser.compress_sequential(features, type)
        for id in ids:
            # self.features_list.append(self.ligth_features_list[id]) # test speed without using facenet
            self.features_list.append(encoding_new(self.get_patches()[id], *(FaceEncoder.encoder)).squeeze())

    def get_features_list(self):
        if not len(self.features_list):
            self.compute_features_list()
        return self.features_list


class Identity():
    def __init__(self, tracklet):
        self.tracklets = [tracklet]
        self.id = None

    def get_features_list(self):
        features_list = []
        for tracklet in self.tracklets:
            features_list += tracklet.get_features_list()
        return features_list

    def set_id(self, id):
        self.id = id
        for tracklet in self.tracklets:
            tracklet.set_id(id)

    def get_trackids(self):
        trackids = []
        for tracklet in self.tracklets:
            trackids.append(tracklet.get_trackid())
        return trackids

    def add_tracklet(self, tracklet):
        self.tracklets.append(tracklet)

    def add_identity(self, identity):
        identity.set_id(self.id)
        for tracklet in identity.tracklets:
            self.tracklets.append(tracklet)

    def merge_identity(self, identity):
        for tracklet in identity.tracklets:
            if tracklet not in self.tracklets:
                self.add_tracklet(tracklet)

    def get_tail(self):
        tail = self.tracklets[0]
        for tracklet in self.tracklets:
            if tail.get_stop_detection() < tracklet.get_stop_detection():
                tail = tracklet
        return tail

    def get_head(self):
        head = self.tracklets[0]
        for tracklet in self.tracklets:
            if head.get_start_detection() > tracklet.get_start_detection():
                head = tracklet
        return head

    def compare_appearance(self, other):
        features_self = self.get_features_list()[-5:]
        features_other = other.get_features_list()[:5]
        return average_compare(np.array(features_self), np.array(features_other))

    def compare_motion(self, other):
        return self.get_tail().compare_motion(other.get_head())

    def compare(self, other):
        # TODO
        track_affinity = 0
        for tracklet1 in self.tracklets:
            for tracklet2 in other.tracklets:
                if tracklet1.get_trackid() == tracklet2.get_trackid():
                    track_affinity = 1

        appearance_affinity = self.compare_appearance(other)
        motion_affinity = self.compare_motion(other)
        logger.debug('motion affinity between {} and {} is: {}'.format(self.get_trackids(), other.get_trackids(),
                                                                       motion_affinity))
        logger.debug('appearance affinity between {} and {} is: {} with {} and {}'.format(self.get_trackids(),
                                                                                          other.get_trackids(),
                                                                                          appearance_affinity,
                                                                                          len(self.get_features_list()),
                                                                                          len(
                                                                                              other.get_features_list())))

        return appearance_affinity + motion_affinity * 0.2 - track_affinity * 0.2

    def overlaps_with(self, identity):
        last_tracklet = self.get_tail()
        first_tracklet = identity.get_head()
        return first_tracklet.overlaps_with(last_tracklet)


class Matcher():

    def __init__(self):
        self.count_id = 0
        self.count_discarded = 0
        self.list_discarded = []
        self.gallery = []  # list of identities
        self.identities_in_batch = []  # list of identities in current batch
        self.tentative_tracklets = []  # tentative tracklets from last batch
        self.confirmed_tracklets = []  # confirmed tracklets in current batch
        self.min_detections_batch = 2
        self.min_detections_track = 5

    def filter_and_add_tracklets(self, probes):
        # add tentative tracklets to probes
        logger.debug('refine, add, filter')

        current_trackids = [str(probe.get_trackid()) for probe in probes]
        for tracklet in self.confirmed_tracklets:
            if str(tracklet.get_trackid()) not in current_trackids and float(get_score(tracklet.probe[-1])) < 0:
                logger.debug('refine tracklet {}'.format(tracklet.get_trackid()))
                logger.debug("{} {}".format(tracklet.id, type(tracklet.id)))
                tracklet.delete_tail_track()

        confirmed_tracklets = []

        mark = np.zeros(len(probes))
        for i, probe in enumerate(probes):
            for tracklet in self.confirmed_tracklets:
                if tracklet.get_trackid() == probe.get_trackid():
                    probe.set_id(tracklet.id)
                    probe.state = TrackletState.Assigned
                    logger.debug('merge tracklet {}'.format(probe.get_trackid()))
                    tracklet.merge_tracklet(probe)
                    mark[i] = 1
                    confirmed_tracklets.append(tracklet)
        probes = [probe for i, probe in enumerate(probes) if not mark[i]]

        logger.debug('tentative tracklets:')
        for tent_tl in self.tentative_tracklets:
            logger.debug(tent_tl.get_trackid())

        mark = np.zeros(len(probes))
        for tent_tl in self.tentative_tracklets:
            for i, probe in enumerate(probes):
                if tent_tl.get_trackid() == probe.get_trackid():
                    # assert probe.state != TrackletState.Assigned
                    logger.debug('found same trackid in current batch')
                    mark[i] = 1
                    if tent_tl.get_nof_detections() + probe.get_nof_detections() >= self.min_detections_track:
                        logger.debug('confirm tentative tracklet {} to probe {}'.format(tent_tl.get_trackid(),
                                                                                        probe.get_trackid()))
                        # add tent_tl to probes
                        # probes.append(tent_tl)
                        probe.state = TrackletState.Assigned
                        tent_tl.state = TrackletState.Confirmed
                        # TODO: merge tracklets, delete tent_tl
                        tent_tl.merge_tracklet(probe)
                        confirmed_tracklets.append(tent_tl)
                    else:
                        # discard both
                        logger.debug('discard tentative tracklet {} and probe {}'.format(tent_tl.get_trackid(),
                                                                                         probe.get_trackid()))
                        self.count_discarded += 1
                        self.list_discarded.append(tent_tl.get_trackid())
                        tent_tl.state = TrackletState.Discarded
                        tent_tl.set_id(1000 + self.count_discarded)
                        probe.state = TrackletState.Discarded
                        probe.set_id(1000 + self.count_discarded)
                        # probes.remove(probe)
            if tent_tl.state == TrackletState.Tentative:
                logger.debug('cannot find any same trackid in current batch, discard {}'.format(tent_tl.get_trackid()))
                # discard this tracklet
                self.count_discarded += 1
                self.list_discarded.append(tent_tl.get_trackid())
                tent_tl.state = TrackletState.Discarded
                tent_tl.set_id(1000 + self.count_discarded)
        probes = [probe for i, probe in enumerate(probes) if not mark[i]]
        confirmed_tentative_tracklets = [tent_tl for tent_tl in self.tentative_tracklets
                                         if tent_tl.state == TrackletState.Confirmed]

        self.tentative_tracklets = []
        # filter probes
        logger.debug('filter out tentative probes')
        logger.debug('len probes before filtering: {}'.format(len(probes)))
        for probe in probes:
            assert probe.state == TrackletState.Tentative
            if probe.get_nof_detections() < self.min_detections_batch:
                if probe.get_stop_track() != probe.fids[-1]:
                    logger.debug('short tracklet, discard {}'.format(probe.get_trackid()))
                    self.count_discarded += 1
                    self.list_discarded.append(probe.get_trackid())
                    probe.set_id(1000 + self.count_discarded)
                    probe.state = TrackletState.Discarded
                else:
                    logger.debug(
                        'short tracklet due to interuption, add {} to tentative tracklets'.format(probe.get_trackid()))
                    # tentative tracklets will be processed in next batch
                    self.tentative_tracklets.append(probe)
            #                    probes.remove(probe)
            else:
                confirmed_tracklets.append(probe)
                probe.state = TrackletState.Confirmed

        self.confirmed_tracklets = confirmed_tracklets
        logger.debug(
            'add {} confirmed and assigned probes for reference in next batch'.format(len(self.confirmed_tracklets)))
        confirmed_probes = [probe for probe in probes if probe.state == TrackletState.Confirmed]
        probes = confirmed_tentative_tracklets + confirmed_probes
        logger.debug('{} probes for use in match in batch'.format(len(probes)))
        return probes

    def new_match_in_batch_hungarian(self, probes, threshold=1.2):

        logger.debug('Matching in batch')
        self.identities_in_batch = []
        probes = self.filter_and_add_tracklets(probes)

        # compute distance matrix
        logger.debug('Start hungarian in batch')
        cost_matrix = np.full((len(probes), len(probes)), 5.0, dtype=np.float32)
        for i in range(len(probes) - 1):
            for j in range(i + 1, len(probes)):
                if not probes[i].overlaps_with(probes[j]):
                    if probes[i].get_start_detection() < probes[j].get_start_detection():
                        cost_matching = probes[i].compare(probes[j])
                        cost_matrix[i][j] = cost_matching
                    else:
                        cost_matching = probes[j].compare(probes[i])
                        cost_matrix[j][i] = cost_matching

        cost_matrix[cost_matrix > threshold] = 5
        # assign tracklet to tracklet
        from scipy.optimize import linear_sum_assignment
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        mark = np.zeros((len(probes) * 2,), dtype=np.int32)
        trackid_to_identity = dict()
        for i, row_ind in enumerate(row_inds):
            if cost_matrix[row_ind][col_inds[i]] < threshold:
                logger.debug('match in batch: {} and {} with {}'.format(probes[row_ind].get_trackid(),
                                                                        probes[col_inds[i]].get_trackid(),
                                                                        cost_matrix[row_ind][col_inds[i]]))
                if not mark[row_ind] and not mark[col_inds[i]]:
                    logger.debug('create a new identity in batch')
                    identity = Identity(probes[row_ind])
                    identity.add_tracklet(probes[col_inds[i]])
                    self.identities_in_batch.append(identity)
                    mark[row_ind] = 1
                    mark[col_inds[i]] = 1
                    trackid_to_identity[probes[row_ind].get_trackid()] = identity
                    trackid_to_identity[probes[col_inds[i]].get_trackid()] = identity
                elif mark[row_ind] == 1 and mark[col_inds[i]] == 1:
                    # merge 2 identities
                    logger.debug('merge 2 identities in batch')
                    identity1 = trackid_to_identity[probes[row_ind].get_trackid()]
                    identity2 = trackid_to_identity[probes[col_inds[i]].get_trackid()]
                    identity1.merge_identity(identity2)
                elif mark[row_ind] == 1:
                    identity = trackid_to_identity[probes[row_ind].get_trackid()]
                    logger.debug('add {} to extant identity {}'.format(probes[col_inds[i]].get_trackid(),
                                                                       identity.get_trackids()))
                    identity.add_tracklet(probes[col_inds[i]])
                    mark[col_inds[i]] = 1
                    trackid_to_identity[probes[col_inds[i]].get_trackid()] = identity
                elif mark[col_inds[i]] == 1:
                    identity = trackid_to_identity[probes[col_inds[i]].get_trackid()]
                    logger.debug('add {} to extant identity {}'.format(probes[row_ind].get_trackid(),
                                                                       identity.get_trackids()))
                    identity.add_tracklet(probes[row_ind])
                    mark[row_ind] = 1
                    trackid_to_identity[probes[row_ind].get_trackid()] = identity

        for i, probe in enumerate(probes):
            if not mark[i]:
                if probe.get_nof_detections() >= self.min_detections_track:
                    self.identities_in_batch.append(Identity(probe))
                elif probe.get_stop_track() == probe.fids[-1]:
                    self.tentative_tracklets.append(probe)
                    probe.state = TrackletState.Tentative
                    self.confirmed_tracklets.remove(probe)
                else:
                    self.count_discarded += 1
                    self.list_discarded.append(probe.get_trackid())
                    probe.state = TrackletState.Discarded
                    probe.set_id(1000 + self.count_discarded)
                    self.confirmed_tracklets.remove(probe)

    def new_match_hungarian(self, probes, threshold=1.2):

        self.new_match_in_batch_hungarian(probes)

        if len(self.gallery) == 0:
            self.gallery = self.identities_in_batch
            for identity in self.identities_in_batch:
                self.count_id += 1
                identity.set_id(self.count_id)
            return 1
        logger.debug('Start hungarian between gallery and batch')
        cost_matrix = np.full((len(self.identities_in_batch), len(self.gallery)), 5.0, dtype=np.float32)
        for i in range(len(self.identities_in_batch)):
            for j in range(len(self.gallery)):
                if not self.gallery[j].overlaps_with(self.identities_in_batch[i]):
                    cost_matching = self.gallery[j].compare(self.identities_in_batch[i])
                    cost_matrix[i][j] = cost_matching
        mark_p = np.zeros((len(self.identities_in_batch),))
        cost_matrix[cost_matrix > threshold] = 5.0
        from scipy.optimize import linear_sum_assignment
        row_inds, col_inds = linear_sum_assignment(cost_matrix)

        for i, row_ind in enumerate(row_inds):
            if cost_matrix[row_ind][col_inds[i]] < threshold:
                logger.debug('choosen min: {} between gallery {} and identity {}'.format(
                    cost_matrix[row_ind][col_inds[i]], self.gallery[col_inds[i]].get_trackids(),
                    self.identities_in_batch[row_ind].get_trackids()))
                self.gallery[col_inds[i]].add_identity(self.identities_in_batch[row_ind])
            else:
                logger.debug('create new identity')
                self.gallery.append(self.identities_in_batch[row_ind])
                self.count_id += 1
                self.identities_in_batch[row_ind].set_id(self.count_id)
            mark_p[row_ind] = 1
        for id, probe in enumerate(self.identities_in_batch):
            if not mark_p[id]:
                # if (cost_matrix[id] > threshold).all():
                logger.debug('create new identity')
                self.gallery.append(self.identities_in_batch[id])
                self.count_id += 1
                self.identities_in_batch[id].set_id(self.count_id)
                if not (cost_matrix[id] > threshold).all():
                    cost_matching = cost_matrix[id, :]
                    logger.debug('have potential for match with {}'.format(
                        cost_matching[np.where(cost_matching < threshold)[0]]))
                    logger.debug(self.identities_in_batch[id].id)
        logger.debug('track_ids in final gallery')
        for _ in self.gallery:
            logger.debug(_.get_trackids())
        logger.debug('discarded tracklets so far: {}'.format(self.count_discarded))
        logger.debug('discarded tracklets: {}'.format(self.list_discarded))
