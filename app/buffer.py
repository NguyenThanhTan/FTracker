from collections import deque
import numpy as np
from app.settings import BATCH_SIZE


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
                frame_left = batch_tracks[max(i - 1, 0)]
                frame_right = batch_tracks[min(i + 1, len(batch_tracks))]
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
        xmin = int((box_curr[0] + box_prev[0] + box_next[0]) / 3)
        ymin = int((box_curr[1] + box_prev[1] + box_next[1]) / 3)
        xmax = int((box_curr[2] + box_prev[2] + box_next[2]) / 3)
        ymax = int((box_curr[3] + box_prev[3] + box_next[3]) / 3)
        return (xmin, ymin, xmax, ymax)


class Buffer(object):
    def __init__(self):
        self.temp_batchtracks = []
        self.temp_fids = []
        self.temp_bitmaps = []
        self.out_batchtracks = deque(maxlen=BATCH_SIZE * 2)
        self.out_fids = deque(maxlen=BATCH_SIZE * 2)
        self.out_bitmaps = deque(maxlen=BATCH_SIZE * 2)
