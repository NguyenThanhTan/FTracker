from concurrent.futures import ThreadPoolExecutor

from app.frame_stream.frame_stream import InputFrameStream
from app.frame_stream.frame_utils import Frame
from imagezmq import imagezmq


class InZMQ(InputFrameStream):
    def __init__(self, open_port):
        self.image_hub = imagezmq.ImageHub(open_port=open_port)
        self.max_frame = 180
        self.frames = []
        self.meta_data = {}
        self.executor = None
        self.init()

    def receive_frame(self):
        image_id, image = self.image_hub.recv_image()
        self.image_hub.send_reply(b'OK')
        return Frame(image_id, image)

    def init(self):
        self.executor = ThreadPoolExecutor(max_workers=1)
        print('now submit the receiving frame task')
        for i in range(0, self.max_frame):
            print('submit %d' % i)
            future = self.executor.submit(self.receive_frame)
            print('future got')
            self.frames.append(future)
            print('append')
        frame = self.receive_frame()
        height, width, _ = frame.frame_data.shape
        self.meta_data['source_size'] = width, height
        self.meta_data['fps'] = 30

    def get(self, fid):
        if fid < 0:
            return None
        if fid >= self.max_frame:
            return None
        return self.frames[fid].result()

    def get_iter(self):
        for future in self.frames:
            yield future.result()

    def get_meta(self):
        return self.meta_data

    def release(self):
        self.frames = []
        self.executor.shut_down()
