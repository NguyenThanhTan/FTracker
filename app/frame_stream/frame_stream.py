import abc
from typing import List

from app.frame_stream import frame_utils


class FrameStream(abc.ABC):
    @abc.abstractmethod
    def init(self):
        pass

    @abc.abstractmethod
    def release(self):
        pass


class InputFrameStream(FrameStream):

    @abc.abstractmethod
    def get(self, idx) -> frame_utils.Frame:
        pass

    @abc.abstractmethod
    def get_iter(self):
        pass

    @abc.abstractmethod
    def get_meta(self):
        pass


class OutputFrameStream(FrameStream):

    @abc.abstractmethod
    def add(self, track_res: frame_utils.TrackResult):
        pass

    @abc.abstractmethod
    def add_batch(self, track_res: List[frame_utils.TrackResult]):
        pass

    @abc.abstractmethod
    def flush(self):
        pass

    @abc.abstractmethod
    def is_done(self):
        pass


class OutCombinedFrameStream(OutputFrameStream):
    def __init__(self, *args, **kwargs):
        if 'ofs_list' in kwargs:
            self.ofs_list = kwargs['ofs_list']
        else:
            self.ofs_list = [arg for arg in args]
        if len(self.ofs_list) == 0:
            raise Exception()

    def init(self):
        for ofs in self.ofs_list:
            ofs.init()

    def release(self):
        print('cleaned %d ofs ' % len(self.ofs_list))
        for ofs in self.ofs_list:
            ofs.release()

    def add(self, track_res):
        if track_res is None:
            return
        for ofs in self.ofs_list:
            ofs.add(track_res)

    def add_batch(self, frame_list):
        for ofs in self.ofs_list:
            ofs.add_batch(frame_list)

    def flush(self):
        for ofs in self.ofs_list:
            ofs.flush()

    def is_done(self):
        all(ofs.is_done for ofs in self.ofs_list)
