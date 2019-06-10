import os

import cv2

from app.frame_stream.frame_stream import OutputFrameStream
from app.settings import DETECTION_COLOR, TRACK_COLOR
from app.frame_stream import frame_utils


class OutVideo(OutputFrameStream):
    __detection_color = DETECTION_COLOR
    __tracking_color = TRACK_COLOR

    def __init__(self, video_path, codec, fps, width, height, colored=True, *args, **kwargs):
        self.frames = []
        self.four_cc = cv2.VideoWriter_fourcc(*codec)
        self.video_path = video_path
        self.fps = fps
        self.frame_size = (width, height)
        self.colored = colored
        self.writer = cv2.VideoWriter()
        self.init()

    def init(self):
        self.frames = []
        if not self.writer.isOpened():
            self.writer.open(self.video_path, self.four_cc, self.fps, self.frame_size, self.colored)

    def is_done(self):
        return os.path.isfile(self.video_path)

    def release(self):
        self.frames = []
        if self.writer.isOpened():
            self.writer.release()

    def add(self, track_res):
        if track_res is None:
            return
        self.frames.append(track_res)

    def add_batch(self, frame_list):
        self.frames += frame_list

    def flush(self):
        if not self.writer.isOpened():
            raise Exception('Writer at %s closed!' % self.video_path)
        for track_res in self.frames:
            new_track_res = frame_utils.mark_boxes(track_res)
            self.writer.write(new_track_res.frame.frame_data)
        self.frames = []
