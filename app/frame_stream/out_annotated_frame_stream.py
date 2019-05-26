import json
from app.frame_stream.frame_stream import FrameStream


class OutAnnotatedFrameStream(FrameStream):
    def __init__(self, file_path):
        self.frames = []
        self.current_frame = 0
        self.locked = False
        self.file_path = file_path

    def start(self):
        self.frames = []
        self.locked = False
        with open(self.file_path, "w+") as f:
            f.truncate(0)
            f.write('[')

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

    def stop(self):
        self.locked = True
        with open(self.file_path, "a+") as f:
            f.write(']')

    def get_iter(self):
        raise NotImplementedError

    def get_meta(self):
        raise NotImplementedError
