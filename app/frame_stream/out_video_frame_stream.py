import cv2

from app.frame_stream.frame_stream import FrameStream
from app.settings import DETECTION_COLOR, TRACK_COLOR


class OutVideoFrameStream(FrameStream):
    __detection_color = DETECTION_COLOR
    __tracking_color = TRACK_COLOR

    def __init__(self, video_path, codec, fps, width, height, colored=True):
        self.frames = []
        self.current_frame = 0
        self.four_cc = cv2.VideoWriter_fourcc(*codec)
        self.video_path = video_path
        self.fps = fps
        self.frame_size = (width, height)
        self.colored = colored
        self.writer = cv2.VideoWriter()

    def start(self):
        if not self.writer.isOpened():
            self.writer.open(self.video_path, self.four_cc, self.fps, self.frame_size, self.colored)

    def add(self, frame, boxes):
        if frame is None:
            return
        self.frames.append((frame, boxes))

    def add_list(self, frame_list):
        self.frames += frame_list

    def get(self, id):
        return self.frames[id]

    def flush(self):
        if not self.writer.isOpened():
            raise Exception('Write closed!')
        if self.current_frame == len(self.frames):
            return
        print("start flush from %d" % self.current_frame)
        for _, ((fid, frame), boxes) in enumerate(self.frames[self.current_frame:]):
            for bid, det in boxes.items():
                if int(bid) > 1000:
                    continue
                cv2.rectangle(frame, (det[0], det[1]), (det[0] + det[2], det[1] + det[3]),
                              self.__detection_color if det[4] > 0 else self.__tracking_color, 2)

                # cv2.putText(frame, "{:.3f}".format(det[4]), (det[0], det[1] + 20),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5, self.__detection_color if det[4] > 0 else self.__tracking_color, 1, cv2.LINE_AA)

                cv2.putText(frame, "{}".format(bid), (int(det[0]), int(det[1] + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, self.__detection_color if det[4] > 0 else self.__tracking_color, 1, cv2.LINE_AA)

                cv2.putText(frame, "Frame id: {}".format(fid), (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, self.__detection_color, 1, cv2.LINE_AA)
            cv2.imwrite("output/office3/{}.jpg".format(fid), frame)
            self.writer.write(frame)
        self.current_frame = len(self.frames)

    def stop(self):
        if self.writer.isOpened():
            self.writer.release()

    def get_iter(self):
        raise NotImplementedError

    def get_meta(self):
        raise NotImplementedError
