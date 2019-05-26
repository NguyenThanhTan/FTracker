from app.frame_stream.frame_stream import FrameStream
import imageio
import numpy as np


class InVideoFrameStream(FrameStream):
    def __init__(self, video_path, fr=0, to=None):
        self.reader = imageio.get_reader(video_path)
        self.size = self.reader.count_frames()
        self.fr = fr
        if to is not None:
            self.to = to

    def add(self, frame, boxes):
        raise NotImplementedError

    def add_list(self, frame_list):
        raise NotImplementedError

    @staticmethod
    def flip_chan(frame):
        r = frame[:, :, 1]
        frame[:, :, 1] = frame[:, :, 2]
        frame[:, :, 2] = frame[:, :, 0]
        frame[:, :, 0] = r
        return frame

    def get(self, fid):
        if fid < 0:
            return None
        rid = fid + self.fr
        if self.to is not None and rid > self.to:
            return None
        return rid, self.flip_chan(self.reader.get_data(rid))

    def get_iter(self):
        i = self.fr
        last = self.reader.count_frames()
        if self.to is not None and self.to + 1 < last:
            last = self.to + 1
        for i in range(i, last):
            yield (i - self.fr, self.flip_chan(self.reader.get_data(i)))
            i += 1

    def get_meta(self):
        return self.reader.get_meta_data()

    def flush(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
