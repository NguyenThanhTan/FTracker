import os

import imageio

if __name__ == '__main__':
    vids = [
        # {"vid": "mobiface80/test/UaSUTyq_raA.mp4", "start": 3, "stop": 900},
        {"vid": "mobiface80/test/U-FP7UU8C58.mp4", "start": 1974, "stop": 3600},
        {"vid": "mobiface80/test/uIdug5IkkaQ.mp4", "start": 22201, "stop": 24630},
        {"vid": "mobiface80/test/WIJpl3pVtSM.mp4", "start": 7651, "stop": 9086},
        {"vid": "mobiface80/test/xjY_LXWPnLw.mp4", "start": 45814, "stop": 47190},
        {"vid": "mobiface80/test/yW4noWcVLQ8.mp4", "start": 8486, "stop": 9300},
        {"vid": "mobiface80/test/7I5t6BAHSGQ.mp4", "start": 391, "stop": 1482},
        {"vid": "mobiface80/test/h0AAQ5CXnRY.mp4", "start": 100114, "stop": 100770},
        {"vid": "mobiface80/test/H0lp_DSqJTs.mp4", "start": 4450, "stop": 5370},
        {"vid": "mobiface80/test/hsRlJ_3xZUk.mp4", "start": 44, "stop": 1260},
        {"vid": "mobiface80/test/Ss4sWrRPChE.mp4", "start": 50209, "stop": 51030}
    ]
    output_folder = 'mobiface80/test'
    for vid in vids:
        ff = vid['vid'].split('.')[0].split('/')[-1]
        reader = imageio.get_reader(vid['vid'])
        output_place = '{of}/{fn}'.format(of=output_folder, fn=ff)
        if not os.path.exists(output_place):
            os.mkdir(output_place)
        print('created dir %s' % output_place)
        for idx in range(vid['start'], vid['stop'] + 1):
            frame = reader.get_data(idx)
            fn = '{op}/{idx}.jpg'.format(op=output_place, idx=str(idx).zfill(8))
            writer = imageio.get_writer(fn)
            writer.append_data(frame)
            writer.close()
            print('created file %s' % fn)
        os.system('cls||clear')
