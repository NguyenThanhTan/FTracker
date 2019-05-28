import glob
import cv2

from app.frame_stream import OutVideoFrameStream
from app.frame_stream.in_video_frame_stream import InVideoFrameStream
from app.frame_stream.out_annotated_frame_stream import OutAnnotatedFrameStream
from app.processor import Processor

import argparse
import os

parser = argparse.ArgumentParser(description='FTracker')
parser.add_argument('--data_dir', default='./videos')
parser.add_argument('--save_dir', default='./output_videos/')
args = parser.parse_args()

import pandas as pd
meta_data = pd.read_csv('./mobiface80/test.meta.csv')

def single(video_id):
    ifs = InVideoFrameStream(
        video_path=os.path.join(args.data_dir, '{}.mp4'.format(video_id)),
        fr=int(meta_data.loc[meta_data['youtube_video_id'] == '{}'.format(video_id)]['first_frame']),
        to=int(meta_data.loc[meta_data['youtube_video_id'] == '{}'.format(video_id)]['last_frame']),
    )
    width, height = ifs.get_meta()['source_size']
    fps = ifs.reader.get_meta_data()['fps']
    ofs = OutVideoFrameStream(
        video_path=os.path.join(args.save_dir, '{}.mp4'.format(video_id)),
        codec='MJPG',
        fps=fps,
        height=height,
        width=width
    )
    # ofs = OutAnnotatedFrameStream(
    #     file_path='output/out.txt'
    # )
    processor = Processor(ifs, ofs)
    processor.start()

def multiple():
    vids = [
        {"vid": "videos/UaSUTyq_raA.mp4", "start": 3, "stop": 900},
        # {"vid": "videos/U-FP7UU8C58.webm", "start": 1974, "stop": 3600},
        {"vid": "videos/U-FP7UU8C58.mp4", "start": 1974, "stop": 3600},
        # {"vid": "videos/uIdug5IkkaQ.webm", "start": 22201, "stop": 24630},
        {"vid": "videos/uIdug5IkkaQ.mp4", "start": 22201, "stop": 24630},
        {"vid": "videos/WIJpl3pVtSM.mp4", "start": 7651, "stop": 9086},
        {"vid": "videos/xjY_LXWPnLw.mp4", "start": 45814, "stop": 47190},
        # {"vid": "videos/yW4noWcVLQ8.webm", "start": 8486, "stop": 9300},
        {"vid": "videos/yW4noWcVLQ8.mp4", "start": 8486, "stop": 9300},
        {"vid": "videos/7I5t6BAHSGQ.mp4", "start": 391, "stop": 1482},
        # {"vid": "videos/h0AAQ5CXnRY.webm", "start": 100114, "stop": 100770},
        {"vid": "videos/h0AAQ5CXnRY.mp4", "start": 100114, "stop": 100770},
        # {"vid": "videos/H0lp_DSqJTs.webm", "start": 4450, "stop": 5370},
        {"vid": "videos/H0lp_DSqJTs.mp4", "start": 4450, "stop": 5370},
        # {"vid": "videos/hsRlJ_3xZUk.webm", "start": 44, "stop": 1260},
        {"vid": "videos/hsRlJ_3xZUk.mp4", "start": 44, "stop": 1260},
        {"vid": "videos/Ss4sWrRPChE.mp4", "start": 50209, "stop": 51030}
    ]

    for vid in vids:
        fn = vid['vid']
        fr = vid['start']
        to = vid['stop']
        ff = fn.split('.')[0].split('/')[-1]
        ifs = InVideoFrameStream(
            video_path=fn,
            fr=fr,
            to=to
        )
        width, height = ifs.get_meta()['source_size']
        fps = ifs.reader.get_meta_data()['fps']
        ofss = []
        ofs = OutVideoFrameStream(
            video_path='output/office3.mp4',
            codec='MJPG',
            fps=fps,
            height=height,
            width=width
        )
        ofss.append(ofs)
        ofs = OutAnnotatedFrameStream(
            file_path='output/test/{ff}.txt'.format(ff=ff)
        )
        ofss.append(ofs)
        processor = Processor(ifs, ofss)
        processor.start()

if __name__ == '__main__':
    # all_frames = './data/videos/office3/*'
    # frames = glob.glob(all_frames)
    # frames = sorted(frames, key=lambda x: int(x.split('/')[-1].split('_')[0].split('.')[0]))
    # frames = [(int(frame.split('/')[-1].split('_')[0].split('.')[0]), cv2.imread(frame)) for frame in frames]
    # test()
    # processor = Assigner()
    # processor.detect_track_match_frames(frames)
    # single('uIdug5IkkaQ')
    multiple()
    print('wtf')
