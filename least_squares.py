import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

plt.rcParams['text.usetex'] = True


class LeastSquares:
    def __init__(self, data, order, data_split):
        self.Y, self.X = data
        self.order = order
        self.Ptrain, self.Pval, self.Ptest = data_split

    def split_data(self):
        idx = np.random.shuffle(np.arange(len(self.Y)))
        Xshuffled = self.X[idx]
        Yshuffled = self.Y[idx]

        Ntrain = int(self.Ptrain * len(self.Y))
        Nval = int(self.Pval * len(self.Y)) + Ntrain
        Ntrain = int(self.Ptrain * len(self.Y)) + Nval

        self.Xtrain, self.Ytrain = Xshuffled[:Ntrain], Yshuffled[:Ntrain]
        self.Xval, self.Yval = Xshuffled[Ntrain:Nval], Yshuffled[Ntrain:Nval]
        self.Xtest, self.Ytest = Xshuffled[Nval:], Yshuffled[Nval:]

    def regression_matrix(self, x):
        N, states = x.shape
        num_params = 1
        A_list = [np.ones(N)]
        for d in range(1, self.order + 1):
            combinations = combinations_with_replacement(range(states), d)
            for combo in combinations:
                poly_feature = np.ones(N)
                for idx in combo:
                    poly_feature *= x[:, idx]
                A_list.append(poly_feature)
                num_params += 1

        A = np.column_stack(A_list)
        return A, num_params

    def OLS(self, A=None):
        if A is None:
            A, num_params = self.regression_matrix(self.X)
        self.theta_hat = np.linalg.pinv(A) @ self.Y
        self.y_hat = A @ self.theta_hat

    def plot_regression(self, lims=(-0.2, 0.8), scatter=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if not scatter:
            ax.plot(self.X[:, 0], self.X[:, 1], self.y_hat, c='r', lw=1.3, label='OLS')
            ax.plot(self.X[:, 0], self.X[:, 1], self.Y, c='k', lw=1.3, label='Measured')
        else:
            ax.scatter(self.X[:, 0], self.X[:, 1], self.Y, c='k', label='Measured')
            ax.scatter(self.X[:, 0], self.X[:, 1], self.y_hat, c='r', label='OLS')
        ax.set_xlabel(r'$\alpha$ [rad]')
        ax.set_ylabel(r'$\beta$ [rad]')
        ax.set_zlabel(r'$C_m$ [-]')
        ax.set_zlim(lims)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def MSE(self):
        return np.mean((self.Y - self.y_hat) ** 2)


def order_influence(data):
    orders = np.arange(1, 21)
    MSEs = []
    for order in orders:
        LS = LeastSquares(data=data, order=order, data_split=(0.7, 0.15, 0.15))
        LS.OLS()
        MSEs.append(LS.MSE())

    plt.plot(orders, MSEs, marker='o')
    plt.xlabel('Order')
    plt.ylabel('MSE')
    plt.grid()
    plt.tight_layout()
    plt.show()
