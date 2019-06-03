import time

from app.frame_stream import frame_utils
from app.frame_stream.frame_stream import OutputFrameStream
from imagezmq import imagezmq

from threading import Thread
import queue as Queue


class TaskQueue(Queue.Queue):
    def __init__(self, num_workers=1):
        Queue.Queue.__init__(self)
        self.num_workers = num_workers
        self.start_workers()

    def add_task(self, task, *args, **kwargs):
        args = args or ()
        kwargs = kwargs or {}
        self.put((task, args, kwargs))

    def start_workers(self):
        for i in range(self.num_workers):
            t = Thread(target=self.worker)
            t.daemon = True
            t.start()

    def worker(self):
        while True:
            item, args, kwargs = self.get()
            item(*args, **kwargs)
            self.task_done()


class OutZMQ(OutputFrameStream):
    def __init__(self, connect_to='tcp://192.168.0.1:5555', *args, **kwargs):
        self.frames = []
        self.connect_to = connect_to
        self.stopped = False
        self.q = TaskQueue(num_workers=1)
        self.sender = None
        self.init()

    def init(self):
        self.frames = []
        self.sender = imagezmq.ImageSender(connect_to='tcp://192.168.0.1:5555')
        self.stopped = False

    def release(self):
        self.q.join()
        self.frames = []
        self.sender = None
        self.stopped = True

    def add(self, track_res):
        if track_res is None:
            return
        self.frames.append(track_res)

    def add_batch(self, frame_list):
        self.frames += frame_list

    def flush_frame(self, track_res):
        image_window_name = 'From Sender'
        new_track_res = frame_utils.mark_boxes(track_res)
        self.sender.send_image(image_window_name, new_track_res.frame.frame_data)
        time.sleep(1 / 5)

    def flush(self):
        for track_res in self.frames:
            self.q.add_task(self.flush_frame, track_res=track_res)
            # self.flush_frame(track_res=track_res)
        self.frames = []
