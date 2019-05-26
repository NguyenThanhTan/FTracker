import glob
import cv2

from app.frame_stream import OutVideoFrameStream
from app.frame_stream.in_video_frame_stream import InVideoFrameStream
from app.frame_stream.out_annotated_frame_stream import OutAnnotatedFrameStream
from app.processor import Processor

if __name__ == '__main__':
    # all_frames = './data/videos/office3/*'
    # frames = glob.glob(all_frames)
    # frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    # frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    # test()
    # processor = Assigner()
    # processor.detect_track_match_frames(frames)
    ifs = InVideoFrameStream(
        video_path='in.mp4',
        fr=55111,
        to=55211
    )
    width, height = ifs.get_meta()['source_size']

    # ofs = OutVideoFrameStream(
    #     video_path='output/office3.mp4',
    #     codec='MJPG',
    #     fps=30,
    #     height=height,
    #     width=width
    # )
    ofs = OutAnnotatedFrameStream(
        file_path='output/out.txt'
    )
    processor = Processor(ifs, ofs)
    processor.start()
    print('wtf')
