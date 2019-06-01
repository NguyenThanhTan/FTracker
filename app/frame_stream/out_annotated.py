import json
import os

from app.frame_stream.frame_stream import OutputFrameStream


class OutAnnotated(OutputFrameStream):
    def __init__(self, file_path):
        self.frames = []
        self.current_frame = 0
        self.locked = False
        self.file_path = file_path
        self.frames = []
        self.locked = False
        self.init()

    def init(self):
        self.frames = []
        with open(self.file_path, "w+") as f:
            f.truncate(0)
            f.write('[')

    def add(self, track_res):
        if track_res is None:
            return
        if self.locked:
            raise Exception("Stream locked!")
        self.frames.append(track_res)

    def add_batch(self, frame_list):
        if self.locked:
            raise Exception("Stream locked!")
        self.frames += frame_list

    def get(self, fid):
        raise NotImplementedError

    def flush(self):
        # if self.current_frame == len(self.frames):
        #     return
        with open(self.file_path, "a+") as f:
            for i, ((fid, frame), boxes) in enumerate(self.frames):
                boxes_r = {}
                for bid, det in boxes.items():
                    boxes_r[bid] = det[:4]
                record = {
                    'fid': fid,
                    'boxes': boxes_r
                }
                f.write(json.dumps(record))
                f.write(',')
            self.frames = []
            # self.current_frame = len(self.frames)
        return

    def release(self):
        self.locked = True
        self.frames = []
        with open(self.file_path, 'rb+') as f:
            f.seek(-1, os.SEEK_END)
            f.truncate()
        with open(self.file_path, "a+") as f:
            f.write(']')
