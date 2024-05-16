import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import pickle
from tqdm import tqdm

torch.random.manual_seed(1)
plt.rcParams['text.usetex'] = True


class Linear(nn.Module):
    """Linear layer of RBF network, containing only the weight matrix as parameter."""

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight)  # calculate the forward pass of the linear layer (y = Wx)


class RBFLayer(nn.Module):
    """RBF layer of the network, containing only the weight matrix as parameter, but dependent on the RBF centers."""

    def __init__(self, in_features, out_features, centers):
        super(RBFLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.centers = centers

    def forward(self, x):
        """Calculate the forward pass of the RBF layer: vj = ∑(wij^2 * (xi - cij)^2)"""
        c = torch.tensor(self.centers, dtype=torch.float32)
        X = torch.zeros(x.shape[0], c.shape[0], x.shape[1])
        for i in range(x.shape[1]):
            xx = x[:, i].reshape(-1, 1)
            cc = c[:, i].reshape(-1, 1).t()
            X[:, :, i] = (xx - cc).pow(2)
        self.distance = X
        vj = sum(self.weight[i, :] ** 2 * X[:, :, i] for i in range(x.shape[1]))
        return vj


class Net(nn.Module):
    """RBF network architecture, containing an RBF layer and a linear layer. Structure depends on number of hidden neurons and RBF
    amplitude."""

    def __init__(self, in_features, hidden_dim, out_features, centers, a):
        super(Net, self).__init__()
        self.layer1 = RBFLayer(in_features, hidden_dim, centers)  # RBF layer connecting input to hidden neurons
        self.layer2 = Linear(hidden_dim, out_features)  # Linear layer connecting hidden neurons to output
        self.a = torch.tensor([a], dtype=torch.float32)  # RBF amplitude

    def forward(self, x):
        """Calculate the forward pass of the network by passing the output of the RBF layer through the RBF exponential activation
        function, the pass that through the linear layer."""
        x = self.a * torch.exp(-self.layer1(x))
        x = self.layer2(x)
        return x


class RBFNet:
    """Class to hold and train the RBF network.
    :param X: input data
    :param Y: output data
    :param hidden_dim: number of hidden neurons
    :param a: RBF amplitude
    :param save: whether to save the trained model or not"""

    def __init__(self, X, Y, hidden_dim, a, save=True):
        self.init_data(X, Y)
        self.hidden_dim = hidden_dim
        self.centers = self.get_centers(self.X_train_tensor, plot=False)  # calculate the RBF centers
        self.a = a
        self.model = Net(3, hidden_dim, 1, self.centers, self.a)  # initialize the RBF network
        self.save = save

    def init_data(self, x, y):
        # split dataset into training and validation sets (50/50)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)
        # normalize the inputs to the network for the different datasets
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
        self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
        self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
        self.X_test = scaler.transform(x)
        self.X_test_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

    def get_centers(self, x, plot=False):
        """Perform KMeans clustering on the input data, with the number of clusters equal to the number of hidden neurons."""
        x = x.detach().numpy()
        kmeans = KMeans(n_clusters=self.hidden_dim, random_state=0).fit(x)  # Kmeans clustering
        centers = kmeans.cluster_centers_   # RBF centers equal to the cluster centers
        labels = kmeans.labels_

        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=25, azim=-110)
            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels, s=0.5, label='Input data', alpha=0.6)
            ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='r', label='Centers', s=20)
            ax.set_title('KMeans Clustering of input', y=0.96)
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(r'$\beta$')
            ax.set_zlabel(r'$V$')
            ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
            plt.tight_layout()
            plt.savefig('plots/kmeans.png', dpi=300)
            plt.show()

        return centers

    def train_lin_reg(self):
        """Train the RBF network using linear regression to solve for the output weights"""
        params = list(self.model.parameters())
        # only update the output weights, not the weights of the RBF layer
        updates = [params[0],
                   torch.linalg.pinv(self.a * torch.exp(-self.model.layer1(self.X_train_tensor))) @ self.y_train_tensor]
        for idx, p in enumerate(self.model.parameters()):
            p.data = updates[idx]

        # evaluate the model and calculate the predicted outputs and losses for the different datasets
        self.model.eval()
        with torch.no_grad():
            self.predictions = [self.model(self.X_train_tensor),
                                self.model(self.X_val_tensor),
                                self.model(self.X_test_tensor)]
            self.losses = [nn.MSELoss()(self.predictions[0], self.y_train_tensor),
                           nn.MSELoss()(self.predictions[1], self.y_val_tensor),
                           nn.MSELoss()(self.predictions[2], self.y_test_tensor)]

        if self.save:
            pickle.dump(self, open('data/rbf_lr_model.pkl', 'wb'))  # save the trained model for later use

    def plot_lin_reg(self):
        """Plot the predicted outputs of the RBF network on the different datasets"""
        x_train, x_val, x_test = self.X_train_tensor.detach().numpy(), self.X_val_tensor.detach().numpy(), self.X_test_tensor.detach().numpy()
        y_train, y_val, y_test = self.y_train_tensor.detach().numpy(), self.y_val_tensor.detach().numpy(), self.y_test_tensor.detach().numpy()

        def plot_data(x, y, idx, title, filename):
            tri = Delaunay(x[:, :2])
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.view_init(elev=30, azim=70)
            ax.scatter(x[:, 0], x[:, 1], y.reshape(-1), c='k', s=0.3, label='Measurements')
            ax.plot_trisurf(x[:, 0], x[:, 1], self.predictions[idx].reshape(-1), triangles=tri.simplices, cmap='coolwarm',
                            label='RBFNet')
            ax.set_zlim(-0.12, 0)
            ax.set_title(title, y=0.98)
            ax.set_xlabel(r'$\alpha$')
            ax.set_ylabel(r'$\beta$')
            ax.set_zlabel(r'$C_m$')
            ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
            plt.tight_layout()
            plt.savefig(f'plots/rbf_{filename}.png', dpi=300)
            plt.show()

        plot_data(x_train, y_train, 0, r'$C_m(\alpha, \beta)$ - RBFNet on training data', 'train')
        plot_data(x_val, y_val, 1, r'$C_m(\alpha, \beta)$ - RBFNet on validation data', 'val')
        plot_data(x_test, y_test, 2, r'$C_m(\alpha, \beta)$ - RBFNet on input data (lin. reg.)', 'test')

    @staticmethod
    def optimize_lin_reg(x, y):
        """Optimize the RBF amplitude for the RBFNet linear regression model minimizing the MSE on the training data."""
        L = []
        n = 600
        with tqdm(desc='RBF amplitude', total=n) as pbar:
            for a in np.linspace(0.05, 3., n):  # iterate over different amplitudes
                rbf = RBFNet(x, y, 20, a, save=False)  # initialize the RBF network
                rbf.train_lin_reg()  # train the network
                L.append(rbf.losses[0])
                pbar.update(1)

        L = np.array(L)
        minimum = np.argmin(L)  # find amplitude leading to minimum MSE
        print(f'Minimum amplitude: {3 * minimum / n}')
        plt.scatter(np.linspace(0.01, 3.01, n), L, label='Train', s=1)
        plt.scatter(np.linspace(0.01, 3.01, n)[minimum], L[minimum], c='r', label='Minimum')
        plt.legend()
        plt.xlabel('a')
        plt.ylabel('MSE')
        plt.title('Optimization of RBF amplitude (lin. reg.)')
        plt.tight_layout()
        plt.savefig('plots/rbf_lr_optimization.png', dpi=300)
        plt.show()
