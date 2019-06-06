import collections

import cv2

from app import settings

Frame = collections.namedtuple('Frame', 'fid frame_data')
TrackResult = collections.namedtuple('TrackResult', 'frame boxes')


def flip_chan(frame):
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def mark_boxes(track_res,
               detection_color=settings.DETECTION_COLOR,
               tracking_color=settings.TRACK_COLOR):
    frame, boxes = track_res
    fid, frame_data = frame
    for bid, det in boxes.items():
        if int(bid) > 1000:
            continue
        cv2.rectangle(frame_data, (det[0], det[1]), (det[0] + det[2], det[1] + det[3]),
                      detection_color if det[4] > 0 else tracking_color, 2)

        # cv2.putText(frame_data, "{:.3f}".format(det[4]), (det[0], det[1] + 20),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5, detection_color if det[4] > 0 else tracking_color, 1, cv2.LINE_AA)

        cv2.putText(frame_data, "{}".format(bid), (int(det[0]), int(det[1] + 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, detection_color if det[4] > 0 else tracking_color, 1, cv2.LINE_AA)

        cv2.putText(frame_data, "Frame id: {}".format(fid), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, detection_color, 1, cv2.LINE_AA)
    return TrackResult(Frame(fid, frame_data), boxes)
