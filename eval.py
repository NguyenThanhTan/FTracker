import json

from mobiface_toolkit.mobiface.trackers import Tracker
from mobiface_toolkit.mobiface.experiments import ExperimentMobiFace


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


class FTracker(Tracker):
    def __init__(self, name='FTracker'):
        super(FTracker, self).__init__(
            name=name,  # tracker name
        )
        self.current_frame = 0
        self.current_dataset = None
        self.fn = None
        self.res = []
        self.track_target = -1

    def register(self, dataset):
        self.current_dataset = dataset
        self.fn = '{op}/{d}.txt'.format(op='output/test', d='_'.join(self.current_dataset.split('_')[:-1]))
        with open(self.fn, 'r') as f:
            self.res = json.loads(f.read())

    def init(self, image, box):
        if self.current_dataset is None:
            raise NotImplementedError
        self.current_frame = 0
        init_box_pos = {
            'x1': box[0],
            'y1': box[1],
            'x2': box[0] + box[2],
            'y2': box[1] + box[3],
        }
        pos = 0
        boxes = self.res[self.current_frame]['boxes']
        for bid in boxes:
            det = boxes[bid]
            box_pos = {
                'x1': det[0],
                'y1': det[1],
                'x2': det[0] + det[2],
                'y2': det[1] + det[3],
            }
            print(init_box_pos, box_pos)
            x = get_iou(init_box_pos, box_pos)
            if x > pos:
                pos = x
                self.track_target = bid
        # perform your initialisation here
        print('Initialisation done!')

    def update(self, image):
        if self.current_dataset is None:
            raise NotImplementedError
        self.current_frame += 1
        boxes = self.res[self.current_frame]['boxes']
        if self.track_target not in boxes:
            return None
        det = boxes[self.track_target]
        box = [det[0], det[1], det[2], det[3]]

        # perform your tracking in the current frame
        # store the result in 'box'
        return box  # [top_x,top_y, width, height]


if __name__ == '__main__':
    # instantiate a tracker
    tracker = FTracker()

    # setup experiment (validation subset)
    experiment = ExperimentMobiFace(
        root_dir='mobiface80/',  # MOBIFACE80 root directory
        subset='test',  # which subset to evaluate ('all', 'train' or 'test')
        result_dir='results',  # where to store tracking results
        report_dir='reports'  # where to store evaluation reports
    )
    experiment.run(tracker, visualize=True)
    experiment.report(['FTracker'])
