import glob

import cv2

from app.frame_stream.frame_stream import InputFrameStream

from app.frame_stream.frame_utils import Frame


class InSequencesFrameStream(InputFrameStream):
    def __init__(self, path, fr=None, to=None, zfill=None):
        self.path = path
        self.fr = fr
        self.to = to
        self.frames = []
        self.meta_data = {}
        self.zfill = zfill
        self.init()

    def init(self):
        if not (self.fr and self.to and self.zfill):
            frames = glob.glob(self.path)
            frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
            self.frames = [cv2.imread(frame) for frame in frames]
        else:
            frames = ['{}/{}.jpg'.format(self.path, str(x).zfill(self.zfill)) for x in
                      range(self.fr, self.to + 1)]
            self.frames = [cv2.imread(y) for y in frames]
        if len(self.frames) == 0:
            return
        height, width, _ = self.frames[0].shape
        self.meta_data = {
            'source_size': (width, height),
            'fps': 30
        }

    def get(self, fid):
        if fid < 0:
            return None
        rid = fid + self.fr
        if self.to is not None and rid > self.to:
            return None
        return Frame(rid, self.frames[fid])

    def get_iter(self):
        for idx, frame in enumerate(self.frames):
            yield Frame(idx, frame)

    def get_meta(self):
        return self.meta_data

    def release(self):
        self.frames = []
