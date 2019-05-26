import cv2

from app.frame_stream.frame_stream import FrameStream
from app.settings import DETECTION_COLOR, TRACK_COLOR


class OutCachedFrameStream(FrameStream):
    def __init__(self):
        self.frames = []
        self.current_frame = 0
        self.locked = False

    def start(self):
        self.frames = []
        self.locked = False

    def add(self, frame, boxes):
        if frame is None:
            return
        if self.locked:
            raise Exception("Stream locked!")
        self.frames.append((frame, boxes))

    def add_list(self, frame_list):
        if self.locked:
            raise Exception("Stream locked!")
        self.frames += frame_list

    def get(self, fid):
        return self.frames[fid]

    def flush(self):
        return

    def stop(self):
        self.locked = True

    def get_iter(self):
        raise NotImplementedError

    def get_meta(self):
        raise NotImplementedError
