import cv2
import numpy as np

from app.buffer import Buffer, Batch
from app.settings import logger, BATCH_SIZE, MATCHING, DETECTION_COLOR, TRACK_COLOR, DETECTOR_MODEL_PATH
from detections.faceboxes.detector import Detector as Fb_detector
from app.face_tracker import FaceTracker
from matching.face_embedding.face_matching import main, Matcher
from copy import deepcopy
from itertools import islice


class Assigner:
    def __init__(self):
        # self.face_detector = SFD_detector(model='detections/sfd/epoch_204.pth.tar')
        self.face_detector = Fb_detector(model_path=DETECTOR_MODEL_PATH)
        self.face_tracker = FaceTracker()
        # self.encoder = FaceEncoder()
        self.matcher = Matcher()

    def draw_frame(self, frames, fid, boxes, writer):
        frame = frames[fid - 1][1]
        for id, det in boxes.items():
            if int(id) > 1000:
                continue
            cv2.rectangle(frame, (det[0], det[1]), (det[0] + det[2], det[1] + det[3]),
                          DETECTION_COLOR if det[4] > 0 else TRACK_COLOR, 2)

            # cv2.putText(frame, "{:.3f}".format(det[4]), (det[0], det[1] + 20),
            #         cv2.FONT_HERSHEY_SIMPLEX,
            #         0.5, DETECTION_COLOR if det[4] > 0 else TRACK_COLOR, 1, cv2.LINE_AA)

            cv2.putText(frame, "{}".format(id), (int(det[0]), int(det[1] + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, DETECTION_COLOR if det[4] > 0 else TRACK_COLOR, 1, cv2.LINE_AA)

            cv2.putText(frame, "Frame id: {}".format(fid), (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, DETECTION_COLOR, 1, cv2.LINE_AA)
        cv2.imwrite("output/office3/{}.jpg".format(fid), frame)
        writer.write(frame)

    def detect_track_match_frames(self, all_frames):
        print('Start process')
        buffer = Buffer()
        batch = Batch(BATCH_SIZE)
        flag = 0

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out_video = 'output/office3.mp4'
        height, width, _ = all_frames[0][1].shape
        writer = cv2.VideoWriter(out_video, fourcc, 30, (width, height), True)
        batch_tracks = []
        for (frame_id, frame) in all_frames:
            height, width, _ = frame.shape
            batch.update(frame, frame_id, height, width)
            detection = self.face_detector.infer(frame, path=False)  # xmin, ymin, xmax, ymax

            logger.debug('Frame {} have {} detections'.format(frame_id, len(detection)))
            detection = np.clip(detection, a_min=0, a_max=None)
            self.face_tracker.match_detection_tracking_hungarian(frame, detection)
            results = self.face_tracker.get_active_track()
            list_tracks = {}
            for box in results['geometries']:
                id = box['id']
                xmin = box['xmin']
                ymin = box['ymin']
                width = box['width']
                height = box['height']
                score = box['score']
                list_tracks[id] = (xmin, ymin, width, height, score)
            batch_tracks.append(list_tracks)
            if batch.count % BATCH_SIZE == 0:

                if MATCHING:
                    if batch.count % (BATCH_SIZE * 2) == 0:
                        flag = 0
                        print('Start matching')
                        batch_tracks = buffer.temp_batchtracks + batch_tracks

                        fids = list(buffer.temp_fids) + list(batch.fids)
                        bitmaps = list(buffer.temp_bitmaps) + list(batch.bitmaps)
                        main(self.matcher, batch_tracks, bitmaps, fids)

                        for idx, tracks in zip(buffer.out_fids, buffer.out_batchtracks):
                            self.draw_frame(all_frames, idx, tracks, writer)

                        buffer.out_batchtracks += batch_tracks[BATCH_SIZE:]
                        buffer.out_fids += fids[BATCH_SIZE:]
                        buffer.out_bitmaps += bitmaps[BATCH_SIZE:]
                    else:
                        flag = 1
                        print('Save batch')
                        buffer.temp_batchtracks = deepcopy(batch_tracks)  # test this without deepcopy
                        buffer.temp_fids = deepcopy(list(batch.fids))
                        buffer.temp_bitmaps = deepcopy(list(batch.bitmaps))

                        print('Save out')
                        buffer.out_batchtracks += buffer.temp_batchtracks
                        buffer.out_fids += buffer.temp_fids
                        buffer.out_bitmaps += buffer.temp_bitmaps
                else:
                    for idx, tracks in zip(batch.fids, batch_tracks):
                        self.draw_frame(all_frames, idx, tracks, writer)

                batch_tracks = []

        # flush the queue -> process the remaining frames
        num_last = batch.count % BATCH_SIZE
        if num_last > 0:
            last_ids = list(islice(batch.fids, 0, BATCH_SIZE))[-num_last:]
            last_bitmaps = list(islice(batch.bitmaps, 0, BATCH_SIZE))[-num_last:]
            last_tracks = batch_tracks[-num_last:]
            if MATCHING:
                if flag == 0:
                    main(self.matcher, last_tracks, last_bitmaps, last_ids)
                    for idx, tracks in zip([buffer.out_fids[i] for i in range(BATCH_SIZE, BATCH_SIZE * 2)],
                                           [buffer.out_batchtracks[i] for i in range(BATCH_SIZE, BATCH_SIZE * 2)]):
                        self.draw_frame(all_frames, idx, tracks, writer)

                    for idx, tracks in zip(last_ids, last_tracks):
                        self.draw_frame(all_frames, idx, tracks, writer)
                else:
                    last_ids = list(buffer.temp_fids) + last_ids
                    last_bitmaps = list(buffer.temp_bitmaps) + last_bitmaps
                    last_tracks = buffer.temp_batchtracks + last_tracks
                    main(self.matcher, last_tracks, last_bitmaps, last_ids)

                    for idx, tracks in zip([buffer.out_fids[i] for i in range(0, BATCH_SIZE)],
                                           [buffer.out_batchtracks[i] for i in range(0, BATCH_SIZE)]):
                        self.draw_frame(all_frames, idx, tracks, writer)

                    for idx, tracks in zip(last_ids, last_tracks):
                        self.draw_frame(all_frames, idx, tracks, writer)
            else:
                for idx, tracks in zip(last_ids, last_tracks):
                    self.draw_frame(all_frames, idx, tracks, writer)
        writer.release()
