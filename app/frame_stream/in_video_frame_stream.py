from app.frame_stream.frame_stream import FrameStream
import imageio
import numpy as np


class InVideoFrameStream(FrameStream):
    def __init__(self, video_path):
        self.reader = imageio.get_reader(video_path)

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

    def get(self, id):
        if id < 0:
            return None
        return id, self.flip_chan(self.reader.get_data(id))

    def get_iter(self):
        i = 0
        for frame in self.reader:
            yield (i, self.flip_chan(frame))
            i += 1

    def get_meta(self):
        return self.reader.get_meta_data()

    def flush(self):
        raise NotImplementedError

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError
