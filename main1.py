"""
AE4320 Assigment: Neural Networks
Author: José Cunha (5216087)
Part 1: State & Parameter estimation with F-16 flight data
"""

import numpy as np
from read_data import train_data, treat_data
from kalman import KalmanFilter
from least_squares import LeastSquares, order_influence

np.random.seed(0)

data = train_data()  # load training data

""" Apply Kalman Filter. For further details into the implementation, please check kalman.py """
# KF = KalmanFilter(dt=0.01, data=data, n_states=4)  # initialize object
# KF.prove_convergence()  # check rank of observability matrix
# KF.IEKF()  # perform Iterative Extended Kalman Filter
# KF.plot()  # plot results

""" Apply Least Squares Estimation. For further details into the implementation, please check least_squares.py """
ols_data = np.loadtxt('data/output.csv', delimiter=',')  # load test data
Y, X = ols_data[:, 0], ols_data[:, 1:]
order = 5

LS = LeastSquares(data=(Y, X), order=order, data_split=(0.7, 0.15, 0.15))  # initialize object
LS.OLS()
LS.plot_regression()

Ytreat, Xtreat = treat_data((Y, X))
LSt = LeastSquares(data=(Ytreat, Xtreat), order=order, data_split=(0.7, 0.15, 0.15))  # initialize object
LSt.OLS()
LSt.plot_regression(lims=(-0.2, 0.2))

order_influence(Ytreat, Xtreat)
