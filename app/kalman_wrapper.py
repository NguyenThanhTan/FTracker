import sys
import numpy as np
from filterpy.kalman import KalmanFilter
import logging


class KalmanTracker:
    def __init__(self, box, img=None, logger=None):
        """
        Init a Kalman tracker given a bounding box (x, y, w, h).  The img param is ignored.
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)

        # Kalman filter, 7-dim state space (x, y, w, r, dx, dy, dw) and 4-dim measurement space (x, y, w, r)
        self.kf = KalmanFilter(dim_x=7, dim_z=4)

        # F matrix: dynamics model (constant velocity and scale change)
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                              [0,1,0,0,0,1,0],
                              [0,0,1,0,0,0,1],
                              [0,0,0,1,0,0,0],
                              [0,0,0,0,1,0,0],
                              [0,0,0,0,0,1,0],
                              [0,0,0,0,0,0,1]])

        # H matrix: conversion from state space to measurement
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                              [0,1,0,0,0,0,0],
                              [0,0,1,0,0,0,0],
                              [0,0,0,1,0,0,0]])

        # P matrix: state covariances
        self.kf.P *= 10.0
        self.kf.P[4:,4:] *= 1000.0

        # R matrix: measurement noise
        self.kf.R[2:,2:] *= 10.0

        # Q matrix: dynamics model noise
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        # filter's init state
        x, y, w, h = box
        if h != 0.0:
            r = w/float(h)
        else:
            r = 0.0
        self.kf.x[:4] = np.array([x, y, w, r]).reshape((-1, 1))

    def get_track_box(self):
        x, y, w, r = self.kf.x[:4].flatten()
        if r != 0.0:
            h = w / r
        else:
            h = 0.0
        return (x, y, w, h)

    def predict(self, img=None):
        """Predict next state and return the predicted bounding box."""

        if self.kf.x[6] + self.kf.x[2] <= 0.0:  # width can't be negative
            self.kf.x[6] = 0.0
        if self.kf.x[3] <= 0.0:  # ratio can't be negative
            self.kf.x[3] = 0.0
        self.kf.predict()
        return self.get_track_box()

    def update(self, box, img=None):
        """Update the filter with new detection box."""
        x, y, w, h = box
        if h != 0.0:
            self.kf.update(np.array([x, y, w, w/float(h)]))
        else:
            self.logger.warning("divided by zero height")