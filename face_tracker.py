import numpy as np
import cv2
import time
import os
from settings import logger
from utils.utilities.utilities import make_folder, to_xywh, cal_iou, to_xyxy
from kalman_wrapper import KalmanTracker
from scipy.optimize import linear_sum_assignment

IOU_THRES = 0.3
KALMAN = 1

class FaceTracks():
    def __init__(self, id, troi, tracker, score, frame):
        self.track_id = id
        self.troi = troi
        self.tracker = tracker
        self.active = True
        self.life_length = 5
        self.hit_strikes = 0
        self.score = score
        self.last_frame = frame

    def cf_predict(self, frame):
        cf_tracker = cv2.TrackerKCF_create()
        cf_tracker.init(self.last_frame, self.troi)
        found, box = cf_tracker.update(frame)
        return found, box

class FaceTracker():
    def __init__(self, allow_live=True):
        self.face_tracks = {}
        self.track_count = -1
        self.n_active_tracks = 0
        self.allow_live = allow_live


    @staticmethod
    def parse(track_id, roi, score):
        """
        Return geometries for each tracked/detected box
        """
        geo = {
                "id": track_id,
                "xmin": int(roi[0]), "ymin": int(roi[1]),
                "width": int(roi[2]), "height": int(roi[3]),
                "score": score
                }
        return geo

    def get_active_track(self):
        track_meta = {"geometries":[]}
        for track in self.face_tracks.values():
            if track.active:
                geo = self.parse(track.track_id, track.troi, track.score)
                track_meta["geometries"].append(geo)
        return track_meta

    def remove_track(self, inactive_id):
        try:
            del self.face_tracks[inactive_id]
            self.n_active_tracks -= 1
        except KeyError:
            logger.error("track id {} does not exist in current track".format(inactive_id))

    def create_track(self, frame, bbox):
        self.track_count += 1
        roi = to_xywh(bbox)
        if KALMAN:
            tracker = KalmanTracker(roi)
            init = 1
        else:
            tracker = cv2.TrackerKCF_create()
            init = tracker.init(frame, roi)
        if not init:
            logger.debug("cannot create tracker for obj [{:.1f}, {:.1f}, {:.1f}, {:.1f}] with id {}".format(bbox[0], bbox[1], bbox[2], bbox[3], self.track_count))
        else:
            logger.debug("create tracker for obj [{:.1f}, {:.1f}, {:.1f}, {:.1f}] with id {}".format(bbox[0], bbox[1], bbox[2], bbox[3], self.track_count))
            self.n_active_tracks += 1
        self.face_tracks[self.track_count] = FaceTracks(self.track_count, roi, tracker, bbox[4], frame)
        # self.face_tracks[self.track_count].last_frame = frame

    def update_track(self, frame, track_id, dbox):
        assert self.face_tracks[track_id].active, "Inactive track cannot be updated"
        if KALMAN:
            update = 1
            self.face_tracks[track_id].tracker.update(to_xywh(dbox))
        else:
            update = self.face_tracks[track_id].tracker.init(frame, to_xywh(dbox))
        self.face_tracks[track_id].last_frame = frame
        if not update:
            logger.debug("cannot update tracker for obj [{:.1f}, {:.1f}, {:.1f}, {:.1f}] with id {}".
                  format(dbox[0], dbox[1], dbox[2], dbox[3], track_id))
        else:
            logger.debug("update box - [{:.1f}, {:.1f}, {:.1f}, {:.1f}] with id {}".format(dbox[0], dbox[1], dbox[2], dbox[3], track_id))
            if self.face_tracks[track_id].life_length < 20:
                self.face_tracks[track_id].life_length += 1

    def match_detection_tracking_hungarian(self, frame, detected_bboxes):
        # TODO: refactor this
        logger.debug('Start Hungarian')
        iou_matrix = np.zeros((len(self.face_tracks), len(detected_bboxes)))
        trois = {track_id: None for track_id in self.face_tracks}
        oks = {track_id: None for track_id in self.face_tracks}
        stopped_tracks = []
        unmatched_tracks = []
        for track_id, track_obj in self.face_tracks.items():
            if not track_obj.active:
                continue
            if KALMAN:
                ok = 1
                troi = track_obj.tracker.predict()
            else:
                ok, troi = track_obj.tracker.update(frame)
            trois[track_id] = troi
            oks[track_id] = ok
            if not ok:
                stopped_tracks.append(track_id)
                # track is lost
                logger.debug("losing track for obj [{:.1f}, {:.1f}, {:.1f}, {:.1f}] with id {:d}".format(
                    track_obj.troi[0],
                    track_obj.troi[1],
                    track_obj.troi[2],
                    track_obj.troi[3],
                    track_id))
                # update track status
                self.face_tracks[track_id].active = False
                # self.remove_track(track_id)
            else:
                for box_id, detected_bbox in enumerate(detected_bboxes):
                    iou_matrix[track_id, box_id] = cal_iou(to_xyxy(troi), detected_bbox)

        cost_matrix = 1 - iou_matrix
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        detected_index = np.zeros((iou_matrix.shape[1]))
        track_index = np.zeros((iou_matrix.shape[0]))
        for i, row_ind in enumerate(row_inds):
            if iou_matrix[row_ind, col_inds[i]] > IOU_THRES:
                self.face_tracks[row_ind].troi = to_xywh(detected_bboxes[col_inds[i]])
                self.face_tracks[row_ind].score = detected_bboxes[col_inds[i]][4]
                self.update_track(frame, row_ind, detected_bboxes[col_inds[i]])
                detected_index[col_inds[i]] = 1
                track_index[row_ind] = 1
            elif oks[row_ind]:
                unmatched_tracks.append(row_ind)

        unmatched_detections = np.where(detected_index == 0)[0]
        iou_matrix = np.zeros((len(unmatched_tracks), len(unmatched_detections)))
        for i, idx_t in enumerate(unmatched_tracks):
            track_obj = self.face_tracks[idx_t]
            found, tbox = track_obj.cf_predict(frame)
            if found:
                for j, idx_d in enumerate(unmatched_detections):
                    iou_matrix[i][j] = cal_iou(to_xyxy(tbox), detected_bboxes[idx_d])
        cost_matrix = 1 - iou_matrix
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        for i, row_ind in enumerate(row_inds):
            if iou_matrix[row_ind, col_inds[i]] > IOU_THRES:
                logger.debug('unmatched track is resurgent {}'.format(unmatched_tracks[row_ind]))
                trackid = unmatched_tracks[row_ind]
                detection_idx = unmatched_detections[col_inds[i]]
                # self.face_tracks[trackid].active = True
                self.face_tracks[trackid].troi = to_xywh(detected_bboxes[detection_idx])
                self.face_tracks[trackid].score = detected_bboxes[detection_idx][4]
                self.update_track(frame, trackid, detected_bboxes[detection_idx])
                detected_index[detection_idx] = 1
                track_index[trackid] = 1

        unmatched_detections = np.where(detected_index == 0)[0]
        iou_matrix = np.zeros((len(stopped_tracks), len(unmatched_detections)))
        for i, idx_t in enumerate(stopped_tracks):
            track_obj = self.face_tracks[idx_t]
            found, tbox = track_obj.cf_predict(frame)
            if found:
                for j, idx_d in enumerate(unmatched_detections):
                    iou_matrix[i][j] = cal_iou(to_xyxy(tbox), detected_bboxes[idx_d])
        cost_matrix = 1 - iou_matrix
        row_inds, col_inds = linear_sum_assignment(cost_matrix)
        for i, row_ind in enumerate(row_inds):
            if iou_matrix[row_ind, col_inds[i]] > IOU_THRES:
                logger.debug('stopped track is resurgent {}'.format(stopped_tracks[row_ind]))
                trackid = stopped_tracks[row_ind]
                detection_idx = unmatched_detections[col_inds[i]]
                self.face_tracks[trackid].active = True
                self.face_tracks[trackid].troi = to_xywh(detected_bboxes[detection_idx])
                self.face_tracks[trackid].score = detected_bboxes[detection_idx][4]
                self.update_track(frame, trackid, detected_bboxes[detection_idx])
                detected_index[detection_idx] = 1
                track_index[trackid] = 1


        for track_id, track_obj in self.face_tracks.items():
            if track_index[track_id] == 0:
                if (not self.allow_live) or (self.face_tracks[track_id].life_length < 0):
                    self.face_tracks[track_id].active = False  # this mean we kill it right away
                    self.face_tracks[track_id].score = -1
                    # self.remove_track(track_id)
                else:
                    self.face_tracks[track_id].troi = trois[track_id]
                    self.face_tracks[track_id].score = -1
                    self.face_tracks[track_id].life_length -= 1
        for box_id, detected_bbox in enumerate(detected_bboxes):
            if detected_index[box_id] == 0:
                self.create_track(frame, detected_bbox)

        # copy_of_dict = dict(self.face_tracks)
        # print(copy_of_dict)
        # for track_id, track_obj in copy_of_dict.items():
        #     if track_obj.active == False:
        #         logger.debug("Trackid {} inactive, detele".format(track_id))
        #         del self.face_tracks[track_id]
        # print(copy_of_dict)
        # print(self.face_tracks)