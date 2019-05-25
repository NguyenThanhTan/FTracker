import glob
import cv2

from frame_stream import StaticVideoFrameStream
from processor import Processor
from detections.sfd.detector import Detector as SFD_detector


def test():
    detector = SFD_detector(model='detections/sfd/epoch_204.pth.tar')
    # extractor = FaceEncoder()
    det = detector.infer(image='./data/videos/office3/0001.jpg')
    frame = cv2.imread('./data/videos/office3/0001.jpg')
    for d in det:
        cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (0, 255, 0), 2)
    cv2.imwrite('test.jpg', frame)

    # all_frames = './data/videos/office3/*'
    # frames = glob.glob(all_frames)
    # frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    # frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    # for idx, frame in frames:
    #     feature = encoding_new(frame, *(FaceEncoder().encoder))
    print('done test')
    # pass


if __name__ == '__main__':
    # all_frames = '/Users/ttnguyen/Documents/work/hades/office3_full/0/*'
    all_frames = './data/videos/office3/*'
    frames = glob.glob(all_frames)
    frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    # test()
    # processor = Assigner()
    # processor.detect_track_match_frames(frames)
    height, width, _ = frames[0][1].shape
    ofs = StaticVideoFrameStream(
        video_path='output/office3.mp4',
        codec='MJPG',
        fps=30,
        height=height,
        width=width
    )
    processor = Processor(None, ofs)
    processor.detect_track_match_frames(frames)
    print('wtf')
