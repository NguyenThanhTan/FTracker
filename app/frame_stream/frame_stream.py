import abc


class FrameStream(abc.ABC):
    @abc.abstractmethod
    def add(self, frame, boxes):
        raise NotImplementedError

    @abc.abstractmethod
    def add_list(self, frame_list):
        raise NotImplementedError

    @abc.abstractmethod
    def get(self, id):
        raise NotImplementedError

    @abc.abstractmethod
    def get_iter(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_meta(self):
        raise NotImplementedError

    @abc.abstractmethod
    def flush(self):
        raise NotImplementedError

    @abc.abstractmethod
    def start(self):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self):
        raise NotImplementedError
