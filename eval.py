from mobiface.trackers import Tracker

class FTracker(Tracker):
    def __init__(self, name = 'FTracker'):
        super(FTracker, self).ace__init__(
            name = name,  # tracker name
        )

    def init(self, image, box):
        # perform your initialisation here
        print('Initialisation done!')


    def update(self, image):
        # perform your tracking in the current frame
        # store the result in 'box'
        return box # [top_x,top_y, width, height]
