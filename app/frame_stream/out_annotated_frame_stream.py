import json

import cv2

from app.frame_stream.frame_stream import FrameStream
from app.settings import DETECTION_COLOR, TRACK_COLOR

class OutAnnotatedFrameStream(FrameStream):
    def __init__(self, file_path):
        self.frames = []
        self.current_frame = 0
        self.locked = False
        self.file_path = file_path

    def start(self):
        self.frames = []
        self.locked = False
        with open(self.file_path, "w") as f:
            f.truncate(0)

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
        # if self.current_frame == len(self.frames):
        #     return
        with open(self.file_path, "w+") as f:
            records = []
            for _, ((fid, frame), boxes) in enumerate(self.frames[self.current_frame:]):
                record = {
                    'fid': fid,
                    'boxes': []
                }
                for bid, det in boxes.items():
                    record['boxes'].append({
                        'bid': bid,
                        'det': det[:3]
                    })
                records.append(record)
            f.write(json.dumps(records))
            # self.current_frame = len(self.frames)
        return

    def stop(self):
        self.locked = True

    def get_iter(self):
        raise NotImplementedError

    def get_meta(self):
        raise NotImplementedError
