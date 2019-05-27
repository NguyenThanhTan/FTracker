import argparse
import pandas as pd
import cv2
import os
import imageio

parser = argparse.ArgumentParser(description='draw ground truth')
parser.add_argument('--gt_dir', default='./mobiface80/test/')
args = parser.parse_args()

meta_data = pd.read_csv('./mobiface80/test.meta.csv')

all_gts = os.listdir(args.gt_dir)
all_gts = [_ for _ in all_gts if '.csv' in _]

vids = [
    {"vid": "UaSUTyq_raA", "start": 3, "stop": 900},
    {"vid": "U-FP7UU8C58", "start": 1974, "stop": 3600},
    {"vid": "uIdug5IkkaQ", "start": 22201, "stop": 24630},
    {"vid": "WIJpl3pVtSM", "start": 7651, "stop": 9086},
    {"vid": "xjY_LXWPnLw", "start": 45814, "stop": 47190},
    {"vid": "yW4noWcVLQ8", "start": 8486, "stop": 9300},
    {"vid": "7I5t6BAHSGQ", "start": 391, "stop": 1482},
    {"vid": "h0AAQ5CXnRY", "start": 100114, "stop": 100770},
    {"vid": "H0lp_DSqJTs", "start": 4450, "stop": 5370},
    {"vid": "hsRlJ_3xZUk", "start": 44, "stop": 1260},
    {"vid": "Ss4sWrRPChE", "start": 50209, "stop": 51030}
]
for vid in vids:
    start = vid['start']
    img_paths = os.listdir(os.path.join(args.gt_dir, vid['vid']))
    img_paths.sort(key=lambda x: int(x.split('.')[0]))
    gt = pd.read_csv(os.path.join(args.gt_dir, vid['vid'] + '_0.annot.csv'))
    count = 0
    for idx, img_path in enumerate(img_paths):
        frame = cv2.imread(os.path.join(args.gt_dir, vid['vid'], img_path))
        det = gt.iloc[idx][1:]
        cv2.rectangle(frame, (det[0], det[1]), (det[0] + det[2], det[1] + det[3]), (255, 0, 255), 2)
        cv2.putText(frame, "Frame id: {}".format(start + idx), (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(args.gt_dir, vid['vid'], img_path), frame)


