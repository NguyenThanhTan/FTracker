from mobiface_toolkit.mobiface.trackers import Tracker
from mobiface_toolkit.mobiface.experiments import ExperimentMobiFace


class FTracker(Tracker):
    def __init__(self, name='FTracker'):
        super(FTracker, self).__init__(
            name=name,  # tracker name
        )

    def init(self, image, box):
        # perform your initialisation here
        print('Initialisation done!')

    def update(self, image):
        # perform your tracking in the current frame
        # store the result in 'box'
        return box  # [top_x,top_y, width, height]


if __name__ == '__main__':
    # instantiate a tracker
    tracker = FTracker()

    # setup experiment (validation subset)
    experiment = ExperimentMobiFace(
        root_dir='/path/to/mobiface80/',  # MOBIFACE80 root directory
        subset='all',  # which subset to evaluate ('all', 'train' or 'test')
        result_dir='results',  # where to store tracking results
        report_dir='reports'  # where to store evaluation reports
    )
    experiment.run(tracker, visualize=True)
