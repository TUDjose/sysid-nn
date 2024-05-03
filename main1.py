"""
AE4320 Assigment: Neural Networks
Author: José Cunha (5216087)
Part 1: State & Parameter estimation with F-16 flight data
"""

import numpy as np
from read_data import train_data
from kalman import KalmanFilter

np.random.seed(0)

data = train_data()  # load training data

""" Apply Kalman Filter. For further details into the implementation, please check kalman.py """
KF = KalmanFilter(dt=0.01, data=data, n_states=4)  # initialize object
KF.prove_convergence()  # check rank of observability matrix
KF.IEKF()  # perform Iterative Extended Kalman Filter
KF.plot()  # plot results
