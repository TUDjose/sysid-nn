"""
AE4320 Assigment: Neural Networks
Author: José Cunha (5216087)
Part 1: State & Parameter estimation with F-16 flight data
"""

import numpy as np
import matplotlib.pyplot as plt
from kalman import KalmanFilter, train_data, validation_data

np.random.seed(0)

# load training data
data = train_data()

# check observability



# do Kalman filtering

KF = KalmanFilter(dt=0.01, data=data)
KF.prove_convergence()
KF.IEKF()
KF.plot()

# reconstruct α_true