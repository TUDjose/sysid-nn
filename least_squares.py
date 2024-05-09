import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from scipy.spatial import Delaunay

plt.rcParams['text.usetex'] = True


class LeastSquares:
    """
    This class contains all the necessary methods to perform an Ordinary Least Squares (OLS) regression on the provided data.
        :param data: tuple containing the data (Y, X)
        :param order: order of the polynomial to fit
    """

    def __init__(self, data, order):
        self.Y, self.X = data
        self.order = order
        self.Xtrain, self.Xval, self.Ytrain, self.Yval = train_test_split(self.X, self.Y, test_size=0.5, random_state=0)
        self.Xtrain, self.Xval = self.Xtrain[:, :2], self.Xval[:, :2]

    def regression_matrix(self, x):
        """Method to create the regression matrix A(x) for the OLS regression, for any odder polynomial, including the cross-coupling
        terms between the states."""
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
        """Perform the actual OLS regression on the provided data from the psuedo-inverse of the regression matrix A and the measured y"""
        if A is None:
            A, num_params = self.regression_matrix(self.Xtrain)
        self.theta_hat = np.linalg.pinv(A) @ self.Ytrain
        self.y_hat = A @ self.theta_hat
        return self.y_hat

    def OLS_predict_on_val_data(self, x=None):
        """Perform OLS regression on a new dataset and configured OLS object."""
        if x is None:
            A, _ = self.regression_matrix(self.Xval)
        else:
            A, _ = self.regression_matrix(x)

        return A @ self.theta_hat

    @staticmethod
    def plot_regression(x, y, yhat, save_file='', title='', elev=25, azim=60, lims=(-0.12, 0.)):
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        tri = Delaunay(x[:, :2])
        ax.scatter(x[:, 0], x[:, 1], y, c='k', label='Measured', s=0.4)
        ax.plot_trisurf(x[:, 0], x[:, 1], yhat,
                        triangles=tri.simplices, label='OLS', cmap='coolwarm', antialiased=True, alpha=1)
        ax.set_xlabel(r'$\alpha$ [rad]')
        ax.set_ylabel(r'$\beta$ [rad]')
        ax.set_zlabel(r'$C_m$ [-]')
        ax.set_zlim(lims)
        ax.legend()
        ax.set_title(title)
        plt.tight_layout()
        if save_file != '': plt.savefig(f'plots/{save_file}.png', dpi=300)
        plt.show()

    def MSE(self, y, yhat):
        """Calculate the Mean Squared Error (MSE) of the OLS regression."""
        return np.mean((y - yhat) ** 2)

    def statistical_validation(self, y_hat_val):
        """Perform statistical validation of the OLS estimation by calculating the variance of the estimated parameters from the
        covariance matrix."""
        epsilon = (self.Yval - y_hat_val).reshape(-1, 1)
        A, _ = self.regression_matrix(self.Xval)
        theta_hat_cov = np.linalg.pinv(A) @ epsilon @ epsilon.T @ np.linalg.pinv(A).T
        variance = np.diag(theta_hat_cov)

        plt.plot(variance, marker='o')
        plt.xlabel('Parameter Index')
        plt.ylabel(r'Var($\hat{\theta}$)')
        plt.title('OLS parameter variance')
        plt.grid()
        plt.tight_layout()
        plt.savefig('plots/OLS_variance.png', dpi=300)
        plt.show()

    def residual_validation(self, y_hat):
        """Perform model-error-based validation of the OLS estimates by calculating the autocorrelation function of the residuals."""
        epsilon = (self.Yval - y_hat)
        A, _ = self.regression_matrix(self.Xval)
        ecorr = np.correlate(epsilon - np.mean(epsilon), epsilon - np.mean(epsilon), mode='full')
        ecorr /= np.max(ecorr)

        plt.plot(np.arange(-len(epsilon), len(epsilon) - 1), ecorr, marker='o', markersize=2)
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'Normalized $K_{\varepsilon\varepsilon}(\tau)$')
        plt.title('OLS Residual Correlation')
        plt.grid()
        plt.tight_layout()
        plt.savefig('plots/OLS_residual_correlation.png', dpi=300)
        plt.show()


def order_influence(data, validation_set):
    """Method to calculate and plot the influence of the order of the polynomial on the MSE of the OLS regression, for the trhee
    different datasets."""
    orders = np.arange(1, 21)
    MSEs = []
    for order in orders:
        ls = LeastSquares(data=data, order=order)
        yhat_train = ls.OLS()
        yhat_val = ls.OLS_predict_on_val_data()
        yhat_special = ls.OLS_predict_on_val_data(x=validation_set[1])
        yhat_full = ls.OLS_predict_on_val_data(x=ls.X[:, :2])
        MSEs.append([ls.MSE(ls.Ytrain, yhat_train),
                     ls.MSE(ls.Yval, yhat_val),
                     ls.MSE(validation_set[0], yhat_special)])

    MSEs = np.array(MSEs)
    plt.figure(figsize=(6, 4))
    plt.semilogy(orders, MSEs, marker='o', label=['Train', 'Validation', 'Special'])
    plt.xlabel('Order')
    plt.ylabel('MSE')
    plt.xticks(np.arange(0, 21, 2))
    plt.grid()
    plt.legend()
    plt.title('Polynomial model order influence')
    plt.tight_layout()
    plt.savefig('plots/order_influence.png', dpi=300)
    plt.show()
