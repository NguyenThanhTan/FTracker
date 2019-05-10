import sys
import os
import glob
import cv2
import numpy as np
from settings import logger
from collections import deque
from detections.sfd.detector import Detector as SFD_detector
from detections.faceboxes.detector import Detector as Fb_detector
from face_tracker import FaceTracker
from matching.face_embedding.face_matching import main, Matcher, Timer
from copy import deepcopy
from itertools import islice

BATCH_SIZE = 32
MATCHING = 1
MIN_SCORE = 0.9
DETECTION_COLOR = (0, 255, 0)
TRACK_COLOR = (0, 0, 255)
class Batch():
    def __init__(self, batch_size):
        self.fids = deque(maxlen=batch_size)
        self.bitmaps = deque(maxlen=batch_size)
        self.metadata = ""
        self.count = 0

    def update(self, frame, frame_id, height, width, metadata=""):
        img_shape = (height, width, 3)
        frame_bitmap = frame
        img = np.frombuffer(frame_bitmap, dtype=np.uint8)
        try:
            img = np.reshape(img, img_shape)
        except:
            print("Frame bytes error. {} - {} - {}".format(height, width, len(frame_bitmap)))
        self.bitmaps.append(img)
        self.fids.append(frame_id)
        self.metadata = metadata
        self.count += 1

    def smooth_tracks(self, batch_tracks):
        smoothed_tracks = []
        for i in range(0, len(batch_tracks)):
            frame_curr = batch_tracks[i]
            try:
                frame_left = batch_tracks[max(i-1, 0)]
                frame_right = batch_tracks[min(i+1, len(batch_tracks))]
                frame_smoothed = {}
                for track_id, box in frame_curr.items():
                    try:
                        box_l = frame_left[track_id]
                        box_r = frame_right[track_id]
                        frame_smoothed[track_id] = self.smooth(box, box_l, box_r)
                        print("smoothed")
                    except KeyError:
                        frame_smoothed[track_id] = box
                        print("cannot smooth")
                smoothed_tracks.append(frame_smoothed)
            except IndexError:
                smoothed_tracks.append(frame_curr)
        return smoothed_tracks

    @staticmethod
    def smooth(box_curr, box_prev, box_next):
        xmin = int((box_curr[0] + box_prev[0] + box_next[0])/3)
        ymin = int((box_curr[1] + box_prev[1] + box_next[1])/3)
        xmax = int((box_curr[2] + box_prev[2] + box_next[2])/3)
        ymax = int((box_curr[3] + box_prev[3] + box_next[3])/3)
        return (xmin,ymin,xmax,ymax)


class Buffer():
    def __init__(self):
        self.temp_batchtracks = []
        self.temp_fids = []
        self.temp_bitmaps = []
        self.out_batchtracks = deque(maxlen=BATCH_SIZE*2)
        self.out_fids = deque(maxlen=BATCH_SIZE*2)
        self.out_bitmaps = deque(maxlen=BATCH_SIZE*2)


class Assigner:

    def __init__(self):
        # self.face_detector = SFD_detector(model='detections/sfd/epoch_204.pth.tar')
        self.face_detector = Fb_detector(model_path='weights/pre_high_pc_rotatelowloss.pth')
        self.face_tracker = FaceTracker()
        # self.encoder = FaceEncoder()
        self.matcher = Matcher()

    def draw_frame(self, frames, fid, boxes, writer):
        frame = frames[fid-1][1]
        for id, det in boxes.items():
            if int(id) > 1000:
                continue
            cv2.rectangle(frame, (det[0], det[1]), (det[0] + det[2], det[1] + det[3]), DETECTION_COLOR if det[4] > 0 else TRACK_COLOR, 2)

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
            detection = self.face_detector.infer(frame, path=False) #xmin, ymin, xmax, ymax

            logger.debug('Frame {} have {} detections'.format(frame_id, len(detection)))
            detection = np.clip(detection, a_min=0, a_max=None)
            self.face_tracker.match_detection_tracking_hungarian(frame, detection)
            results = self.face_tracker.get_active_track()
            list_tracks = {}
            for box in results['geometries']:
                id=box['id']
                xmin=box['xmin']
                ymin=box['ymin']
                width=box['width']
                height=box['height']
                score=box['score']
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
                        buffer.temp_batchtracks = deepcopy(batch_tracks) #test this without deepcopy
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

        ## flush the queue -> process the remaining frames
        num_last = batch.count % BATCH_SIZE
        if num_last > 0:
            last_ids = list(islice(batch.fids, 0, BATCH_SIZE))[-num_last:]
            last_bitmaps = list(islice(batch.bitmaps, 0, BATCH_SIZE))[-num_last:]
            last_tracks = batch_tracks[-num_last:]
            if MATCHING:
                if flag == 0:
                    main(self.matcher, last_tracks, last_bitmaps, last_ids)
                    for idx, tracks in zip([buffer.out_fids[i] for i in range(BATCH_SIZE, BATCH_SIZE*2)],
                                           [buffer.out_batchtracks[i] for i in range(BATCH_SIZE, BATCH_SIZE*2)]):
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

def test():
    detector = SFD_detector(model='detections/sfd/epoch_204.pth.tar')
    # extractor = FaceEncoder()
    det = detector.infer(image='./data/videos/office3/0001.jpg')
    frame = cv2.imread('./data/videos/office3/0001.jpg')
    for d in det:
        cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 2)
    cv2.imwrite('test.jpg', frame)

    # all_frames = './data/videos/office3/*'
    # frames = glob.glob(all_frames)
    # frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    # frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    # for idx, frame in frames:
    #     feature = encoding_new(frame, *(FaceEncoder().encoder))
    print('done test')
    # pass


if __name__ == '__main__':
    # test()
    processor = Assigner()
    # all_frames = '/Users/ttnguyen/Documents/work/hades/office3_full/0/*'
    all_frames = './data/videos/office3/*'
    frames = glob.glob(all_frames)
    frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    processor.detect_track_match_frames(frames)

    print('wtf')
