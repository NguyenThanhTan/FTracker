from app.frame_stream.frame_stream import InputFrameStream
import imageio

from app.frame_stream.frame_utils import flip_chan, Frame


class InVideoFrameStream(InputFrameStream):
    def __init__(self, path, fr=0, to=None):
        self.path = path
        self.fr = fr
        if to is not None:
            self.to = to
        self.reader = None
        self.size = None
        self.init()

    def init(self):
        self.reader = imageio.get_reader(self.path)
        self.size = self.reader.count_frames()

    def get(self, fid):
        if fid < 0:
            return None
        rid = fid + self.fr
        if self.to is not None and rid > self.to:
            return None
        frame_data = flip_chan(self.reader.get_data(rid))
        return Frame(rid, frame_data)

    def get_iter(self):
        i = self.fr
        last = self.reader.count_frames()
        if self.to is not None and self.to + 1 < last:
            last = self.to + 1
        for i in range(i, last):
            yield (i - self.fr, flip_chan(self.reader.get_data(i)))
            i += 1

    def get_meta(self):
        return self.reader.get_meta_data()

    def release(self):
        self.reader.close()
