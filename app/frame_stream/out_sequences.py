import cv2

from app.frame_stream import frame_utils
from app.frame_stream.frame_stream import OutputFrameStream
from app.settings import DETECTION_COLOR, TRACK_COLOR


class OutSequences(OutputFrameStream):
    __detection_color = DETECTION_COLOR
    __tracking_color = TRACK_COLOR

    def __init__(self, sequences_prefix):
        self.frames = []
        self.seq_prefix = sequences_prefix
        self.init()

    def init(self):
        self.frames = []

    def release(self):
        self.frames = []

    def add(self, track_res):
        if track_res is None:
            return
        self.frames.append(track_res)

    def add_batch(self, frame_list):
        self.frames += frame_list

    def flush(self):
        for track_res in self.frames:
            new_track_res = frame_utils.mark_boxes(track_res)
            cv2.imwrite(self.seq_prefix.format(new_track_res.frame.fid), new_track_res.frame.frame_data)
        self.frames = []
