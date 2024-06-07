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
        # Split the data (50/ 50) into training and validation sets
        self.Xtrain, self.Xval, self.Ytrain, self.Yval = train_test_split(self.X, self.Y, test_size=0.5, random_state=0)
        # To allow for the special validation set to be used, only use the first two states (α, β) to estimate Cm
        self.Xtrain, self.Xval = self.Xtrain[:, :2], self.Xval[:, :2]

    def regression_matrix(self, x):
        """Method to create the regression matrix A(x) for the OLS regression, for any odder polynomial, including the cross-coupling
        terms between the states."""
        N, states = x.shape
        num_params = 1
        A_list = [np.ones(N)]
        for d in range(1, self.order + 1):
            combinations = combinations_with_replacement(range(states), d)  # Get all combinations of the states for all orders d
            for combo in combinations:
                poly_feature = np.ones(N)
                for idx in combo:
                    poly_feature *= x[:, idx]
                A_list.append(poly_feature)
                num_params += 1

        A = np.column_stack(A_list)
        return A, num_params

    def OLS(self, A=None):
        """Perform the actual OLS regression on the provided data from the pseudo-inverse of the regression matrix A and the measured y:
        P1.5"""
        if A is None:
            A, num_params = self.regression_matrix(self.Xtrain)     # Get the regression matrix for the training data
        self.theta_hat = np.linalg.pinv(A) @ self.Ytrain    # Calculate the OLS parameters
        self.y_hat = A @ self.theta_hat   # Calculate the OLS estimate
        return self.y_hat

    def OLS_predict_on_val_data(self, x=None):
        """Perform OLS regression on a new dataset and configured OLS object: P1.9"""
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
        ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
        ax.set_title(title, y=0.98)
        plt.tight_layout()
        if save_file != '':
            plt.savefig(f'plots/{save_file}.png', dpi=300)
        plt.show()

    def MSE(self, y, yhat):
        """Calculate the Mean Squared Error (MSE) of the OLS regression."""
        return np.mean((y - yhat) ** 2)

    def statistical_validation(self, y_hat_val):
        """Perform statistical validation of the OLS estimation by calculating the variance of the estimated parameters from the
        covariance matrix: P1.8"""
        epsilon = (self.Yval - y_hat_val).reshape(-1, 1)    # calculate residuals
        A, _ = self.regression_matrix(self.Xval)    # Get the regression matrix for the validation data
        n = len(epsilon)  # Number of data points
        k = A.shape[1]  # Number of parameters
        theta_hat_cov = np.linalg.pinv(A) @ epsilon @ epsilon.T @ np.linalg.pinv(A).T / (n - k)   # OLS parameters covariance matrix

        idxs = np.arange(min(theta_hat_cov.shape))
        x = np.arange(theta_hat_cov.shape[1])
        y = np.arange(theta_hat_cov.shape[0])
        x, y = np.meshgrid(x, y)

        fig = plt.figure(figsize=(7 , 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=20, azim=-105)
        ax.plot_surface(x, y, theta_hat_cov, cmap='coolwarm', label='Covariance surface')
        ax.scatter(idxs, idxs, np.diag(theta_hat_cov), c='k', s=30, label='Variance')
        ax.set_xlabel(r'$\hat{\theta}_i$')
        ax.set_ylabel(r'$\hat{\theta}_j$')
        ax.set_zlabel(r'Cov($\hat{\theta}$)')
        ax.set_title('OLS parameter covariance surface', y=0.95)
        ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
        plt.tight_layout()
        plt.savefig('plots/OLS_variance.png', dpi=300)
        plt.show()


    def residual_validation(self, y_hat):
        """Perform model-error-based validation of the OLS estimates by calculating the autocorrelation function of the residuals: P1.8"""
        epsilon = (self.Yval - y_hat)   # Calculate the residuals
        A, _ = self.regression_matrix(self.Xval)    # Get the regression matrix for the validation data
        ecorr = np.correlate(epsilon - np.mean(epsilon), epsilon - np.mean(epsilon), mode='full')   # Calculate the autocorrelation function
        ecorr /= np.max(ecorr)  # Normalize the autocorrelation function
        ci = 1.96 / np.sqrt(len(epsilon))   # Calculate the 95% confidence interval

        plt.fill_between(np.arange(-len(epsilon), len(epsilon) - 1), -ci, ci, color='red', alpha=0.5, label=r'95\% CI')
        plt.plot(np.arange(-len(epsilon), len(epsilon) - 1), ecorr, marker='o', markersize=1, label='Autocorrelation')
        plt.xlabel(r'$\tau$')
        plt.ylabel(r'Normalized $K_{\varepsilon\varepsilon}(\tau)$')
        plt.title('OLS Residual Correlation')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/OLS_residual_correlation.png', dpi=300)
        plt.show()

    def residual_RMS(self, y, yhat):
        """Calculate the Root Mean Squared (RMS) Error of the OLS regression on the special validation set: P1.9"""
        rms = np.sqrt(np.mean((y - yhat) ** 2))
        print(f"The residual RMS of the OLS model of polynomial order {self.order} is: {rms:.4f}")

    @staticmethod
    def order_influence(data, validation_set):
        """Method to calculate and plot the influence of the order of the polynomial on the MSE of the OLS regression, for the three
        different datasets: P1.7"""
        orders = np.arange(1, 21)
        MSEs = []
        for order in orders:    # Loop over the different orders of the OLS estimator
            ls = LeastSquares(data=data, order=order)
            yhat_train = ls.OLS()   # Perform OLS regression on the training data
            yhat_val = ls.OLS_predict_on_val_data() # Perform OLS regression on the validation data
            yhat_special = ls.OLS_predict_on_val_data(x=validation_set[1]) # Perform OLS regression on the special validation data
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
