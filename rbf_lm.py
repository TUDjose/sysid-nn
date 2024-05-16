import csv
import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from read_data import treat_data, validation_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
import pickle

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
    """RBF layer of the network, containing as parameters the weight matrix and RBF centers"""

    def __init__(self, in_features, out_features, centers):
        super(RBFLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.centers = centers

    def forward(self, x):
        """Calculate the forward pass of the RBF layer: vj = ∑(wij^2 * (xi - cij)^2)"""
        c = self.centers.clone()
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
    amplitude. For this case, the RBF amplitude is also a learnable parameter."""

    def __init__(self, in_features, hidden_dim, out_features, centers):
        super(Net, self).__init__()
        self.layer1 = RBFLayer(in_features, hidden_dim, centers)
        self.layer2 = Linear(hidden_dim, out_features)
        self.a = nn.Parameter(torch.tensor([1.], dtype=torch.float32, requires_grad=True))

    def forward(self, x):
        """Calculate the forward pass of the network by passing the output of the RBF layer through the RBF exponential activation
        function, then pass that through the linear layer."""
        x = self.a * torch.exp(-self.layer1(x))
        x = self.layer2(x)
        return x


class RBFNet:
    """Class to hold and train the RBF network with the Levenberg-Marquardt algorithm. Class  also contains methods for network
    evaluation and validation.
    :param X: input data
    :param Y: output data
    :param hidden_dim: number of hidden neurons
    :param epochs: maximum number of epochs to train
    :param mu: damping factor for LM algorithm
    :param goal: goal value for loss function
    :param input_dim: number of input features (3 or 2)
    :param filename: filename to save the object
    :param save: whether to save the object
    :param retry: whether to retry training automatically if the goal is not reached"""

    def __init__(self, X, Y, hidden_dim, epochs, mu, goal, input_dim=3, filename=None, save=True, retry=False):
        self.X, self.Y = X, Y
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.mu = mu
        self.goal = goal
        self.input_dim = input_dim
        self.filename = f'data/rbf_LM_{hidden_dim}.pkl' if not filename else filename
        self.save = save
        self.retry = retry

        self.init_data(X, Y)
        # initialize RBF centers with kmeans clustering and make them learnable parameters
        self.centers = nn.Parameter(torch.tensor(self.get_centers(self.X_train_tensor), requires_grad=True))
        self.model = Net(input_dim, hidden_dim, 1, self.centers)  # initialize the RBF network

    def init_data(self, x, y):
        # split dataset into training and validation sets (50/50)
        X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.5, random_state=0)

        # normalize the inputs to the network for the different datasets and input sizes
        if self.input_dim == 3:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(x)
            self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
            self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
            self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            self.y_test_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)

        if self.input_dim == 2:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(x)
            y_spec, a_spec, b_spec = validation_data()
            X_spec = np.vstack((a_spec, b_spec)).T
            X_spec = scaler.transform(X_spec)
            self.X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
            self.y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
            self.X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
            self.y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)
            self.X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
            self.y_test_tensor = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
            self.X_spec_tensor = torch.tensor(X_spec, dtype=torch.float32)
            self.y_spec_tensor = torch.tensor(y_spec, dtype=torch.float32).reshape(-1, 1)

    def get_centers(self, x):
        """Perform KMeans clustering on the input data, with the number of clusters equal to the number of hidden neurons."""
        x = x.detach().numpy()
        kmeans = KMeans(n_clusters=self.hidden_dim, random_state=0).fit(x)  # Kmeans clustering
        centers = kmeans.cluster_centers_  # RBF centers equal to the cluster centers
        return centers

    def get_outputs(self):
        """Calculate the outputs of the network for the different datasets as inputs."""
        outputs = {
            'train': self.model(self.X_train_tensor),
            'val': self.model(self.X_val_tensor),
            'test': self.model(self.X_test_tensor),
            'special': self.model(self.X_spec_tensor) if self.input_dim == 2 else None
        }
        return outputs

    def get_losses(self, out):
        """Calculate the residual sum of squares (RSS) and mean squared error (MSE) for the different datasets."""
        RSS = lambda x, y: (x - y).pow(2).sum()     # calculate the residual sum of squares
        MSE = nn.MSELoss()  # calculate the mean squared error
        losses = {
            'train_RSS': RSS(out['train'], self.y_train_tensor).item(),
            'val_RSS': RSS(out['val'], self.y_val_tensor).item(),
            'test_RSS': RSS(out['test'], self.y_test_tensor).item(),
            'train_MSE': MSE(out['train'], self.y_train_tensor).item(),
            'val_MSE': MSE(out['val'], self.y_val_tensor).item(),
            'test_MSE': MSE(out['test'], self.y_test_tensor).item(),
            'special_RSS': RSS(out['special'], self.y_spec_tensor).item() if self.input_dim == 2 else None,
            'special_MSE': MSE(out['special'], self.y_spec_tensor).item() if self.input_dim == 2 else None
        }
        return losses

    def train_lm(self):
        """Train the RBF network using the Levenberg-Marquardt algorithm. The different LMA steps are detailed in this method"""

        def jacobian():
            """Calculate the Jacobian matrix components of the error vector w.r.t. the weights vector"""
            A, W1, C, W2 = list(self.model.parameters())  # get weights/learnable parameters of network
            h = A.item() * torch.exp(-self.model.layer1(self.X_train_tensor))  # output of RBF activation function
            y = self.model(self.X_train_tensor)  # output of the network

            # derivative of error w.r.t. output weights
            dedw2 = -1 * (self.y_train_tensor - y) * h
            # derivative of error w.r.t. RBF weights for the different input features (α, β)
            dedw1a = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[0, :] * self.model.layer1.distance[:, :, 0]
            dedw1b = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[1, :] * self.model.layer1.distance[:, :, 1]
            # derivative of error w.r.t. RBF amplitude
            deda = -1 * (self.y_train_tensor - y) @ W2.T * torch.exp(-self.model.layer1(self.X_train_tensor))
            # derivative of error w.r.t. RBF centers for the different input features (α, β)
            dedCa = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[0, :] ** 2 * self.model.layer1.distance[:, :, 0].pow(0.5)
            dedCb = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[1, :] ** 2 * self.model.layer1.distance[:, :, 1].pow(0.5)

            if self.input_dim == 3:
                # derivative of error w.r.t. RBF weights and RBF centers for the total velocity (V) input
                dedw1V = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[2, :] * self.model.layer1.distance[:, :, 2]
                dedCV = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[2, :] ** 2 * self.model.layer1.distance[:, :, 2].pow(0.5)
                return deda, torch.hstack([dedw1a, dedw1b, dedw1V]), torch.hstack([dedCa, dedCb, dedCV]), dedw2
            else:
                return deda, torch.hstack([dedw1a, dedw1b]), torch.hstack([dedCa, dedCb]), dedw2

        pbar = tqdm(desc='Epochs', unit=' epoch')
        update = True
        mu_inc, mu_dec = 2, 10  # increase and decrease factors for damping factor within the LMA
        self.losses = []  # store the loss values for the different datasets
        self.mus = []  # store the damping factor values

        for epoch in range(self.epochs):  # iterate over the maximum number of epochs
            try:
                self.mus.append(self.mu)
                """1. If network should be updated, calculate the outputs and losses"""
                if update:
                    self.model.zero_grad()  # zero the gradients of the model
                    outputs = self.get_outputs()  # get the outputs of the network
                    loss = self.get_losses(outputs)  # calculate the differnt loss values
                    self.losses.append(loss)
                    E = 0.5 * (self.y_train_tensor - outputs['train']).pow(2)  # calculate the error vector
                    Jacobian = jacobian()  # calculate the Jacobian matrix

                """2. Calculate the LMA weight updates"""
                deltas = []
                for idx, J in enumerate(Jacobian):
                    deltaW = torch.linalg.inv(J.t() @ J + self.mu * torch.eye(J.shape[1])) @ J.t() @ E  # calculate the LMA weight updates
                    deltas.append(deltaW)
                deltas[0] = torch.mean(deltas[0], dim=0)  # amplitude parameter update must be of shape (1,)

                """3. Update the network parameters  on a dummy network and calculate its loss"""
                MODEL = RBFNet(self.X, self.Y, self.hidden_dim, 1, self.mu, self.goal, input_dim=self.input_dim)  # dummy network
                MODEL.model.load_state_dict(self.model.state_dict())  # make the dummy network equal to the current network
                for i, p in enumerate(MODEL.model.parameters()):
                    p.data -= deltas[i].reshape(p.shape)  # update the parameters of the dummy network
                loss_new = MODEL.get_losses(MODEL.get_outputs())  # calculate the loss of the dummy network

                """4. If the dummy loss is lower than the current loss, decrease μ and update the current network parameters
                      If the dummy loss is higher than the current loss, increase μ and do not update the current network parameters"""
                if loss_new['train_RSS'] < self.losses[-1]['train_RSS']:
                    self.mu /= mu_dec  # decrease damping factor
                    update = False
                    for i, p in enumerate(self.model.parameters()):
                        p.data -= deltas[i].reshape(p.shape)  # actually update the current network parameters
                    self.losses.append(self.get_losses(self.get_outputs()))
                else:
                    self.mu *= mu_inc  # increase damping factor
                    update = True
                    # do not update params

                if self.mu >= 1e6:  # if damping factor is too high, stop training, as the network has converged
                    self.epochs = epoch
                    break

                pbar.update(1)
                # print(f'Epoch [{epoch + 1}/{self.epochs}], Train Loss: '
                #       f'{loss["train_RSS"]:.6f}, a: {self.model.a.item():.6f},  mu: {self.mu:.4f}')

            except torch._C._LinAlgError as e:  # if the Jacobian matrix ends up being singular, restart the training automatically
                print(f'LinAlgError: {e}')
                self.train_lm()
                break

            except KeyboardInterrupt:
                break

        pbar.close()
        # if the goal is not reached and retry is enabled, restart the training automatically
        if self.losses[-1]['train_RSS'] > self.goal and self.retry:
            print(f'Failed to converge,  loss={self.losses[-1]["train_RSS"]:.6f}, trying again ...')
            rbf = RBFNet(self.X, self.Y, self.hidden_dim, 500, 0.01, self.goal, retry=True, filename=self.filename,
                         input_dim=self.input_dim)
            rbf.train_lm()
        else:
            # if the goal is reached, evaluate the network on the validation data
            self.model.eval()
            with torch.no_grad():
                self.predictions = self.model(self.X_val_tensor)
                self.outputs = self.get_outputs()  # get the outputs of the network

            print(f'Final Loss (RSS/MSE): {self.losses[-1]["train_RSS"]:.6f}/{self.losses[-1]["train_MSE"]:.6f} at epoch {epoch}')

            if self.save:
                pickle.dump(self, open(self.filename, 'wb'))  # save the object to a file for later use

    def train_lm_static(self):
        """Train the RBF network using the Levenberg-Marquardt weight update, but with a static damping factor."""

        def jacbobian():
            """Same functionality as train_lm.jacobian() -  calculate the Jacobian of error w.r.t. network weights """
            A, W1, C, W2 = list(self.model.parameters())
            h = A.item() * torch.exp(-self.model.layer1(self.X_train_tensor))
            y = self.model(self.X_train_tensor)

            dedw2 = -1 * (self.y_train_tensor - y) * h
            dedw1a = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[0, :] * self.model.layer1.distance[:, :, 0]
            dedw1b = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[1, :] * self.model.layer1.distance[:, :, 1]
            dedw1V = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * 2 * W1[2, :] * self.model.layer1.distance[:, :, 2]
            deda = -1 * (self.y_train_tensor - y) @ W2.T * torch.exp(-self.model.layer1(self.X_train_tensor))
            dedCa = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[0, :] ** 2 * self.model.layer1.distance[:, :, 0].pow(0.5)
            dedCb = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[1, :] ** 2 * self.model.layer1.distance[:, :, 1].pow(0.5)
            dedCV = -1 * (self.y_train_tensor - y) @ W2.T * (-h) * -2 * W1[2, :] ** 2 * self.model.layer1.distance[:, :, 2].pow(0.5)

            return deda, torch.hstack([dedw1a, dedw1b, dedw1V]), torch.hstack([dedCa, dedCb, dedCV]), dedw2

        criterion = lambda x, y: (x - y).pow(2).sum()  # calculate RSS of the network
        self.losses = []
        for epoch in range(self.epochs):    # iterate over the maximum number of epochs
            try:
                self.model.zero_grad()  # zero the gradients of the model
                outputs = self.model(self.X_train_tensor)   # get the outputs of the network using training data as input
                loss = criterion(outputs, self.y_train_tensor)  # calculate the RSS loss of the network
                self.losses.append(loss.item())
                E = 0.5 * (self.y_train_tensor - outputs).pow(2)    # calculate the error vector
                Jacobian = jacbobian()  # calculate the Jacobian matrix

                deltas = []
                for idx, J in enumerate(Jacobian):
                    deltaW = torch.linalg.inv(J.t() @ J + self.mu * torch.eye(J.shape[1])) @ J.t() @ E  # calculate the LMA weight updates
                    deltas.append(deltaW)
                deltas[0] = torch.mean(deltas[0], dim=0)    # amplitude parameter update must be of shape (1,)

                for i, p in enumerate(self.model.parameters()):
                    p.data -= deltas[i].reshape(p.shape)    # update the parameters of the network

                if self.losses[-1] < self.goal:     # if the goal is reached, stop training
                    break

            except KeyboardInterrupt:
                break

        # evaluate the network on the validation data
        self.model.eval()
        with torch.no_grad():
            self.predictions = self.model(self.X_val_tensor)

        if self.save:
            pickle.dump(self, open(self.filename, 'wb'))    # save the object to a file for later use

    def show_mu_dynamics(self):
        """Plot the damping factor changes during training."""
        plt.semilogy(self.mus)
        plt.xlabel('Epoch')
        plt.ylabel(r'$\mu$')
        plt.title('Damping factor changes during training')
        plt.grid()
        plt.tight_layout()
        plt.savefig('plots/mu_dynamics.png', dpi=300)
        plt.show()

    @staticmethod
    def plot_lm(filename, net=None):
        """Plot the RBF network predictions on the input data and the actual measurements."""
        if not net:
            rbf = pickle.load(open(filename, 'rb'))     # load the RBF network object
            x = rbf.X_val_tensor.detach().numpy()    # get the input data
            y = rbf.y_val_tensor.detach().numpy()   # get the measurement data
            p = rbf.predictions.detach().numpy()    # get the network predictions
        else:
            rbf = net
            x = rbf.X_val_tensor.detach().numpy()
            y = rbf.y_val_tensor.detach().numpy()
            p = rbf.predictions.detach().numpy()

        tri = Delaunay(x[:, :2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=33, azim=70)
        ax.scatter(x[:, 0], x[:, 1], y.reshape(-1), c='k', s=0.3, label='Measurements')
        ax.plot_trisurf(x[:, 0], x[:, 1], p.reshape(-1), triangles=tri.simplices, cmap='coolwarm', label='RBFNet')
        ax.set_zlim(-0.12, 0)
        ax.set_title(r'$C_m(\alpha, \beta)$ - RBFNet on input data (LMA)', y=0.98)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.set_zlabel(r'$C_m$')
        ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
        plt.tight_layout()
        plt.savefig(f'plots/rbf_LM.png', dpi=300)
        plt.show()

    @staticmethod
    def damping_effect(x, y, plot=False):
        """Calculate and plot the influence of the damping factor on the convergence of the network. The damping factor is varied and the
        loss curve is plotted for each value of the damping factor to show its effect on convergence."""
        L = []
        mus = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]    # different damping factors to test
        if not plot:
            for mu in mus:  # iterate over the different damping factors
                print(mu)
                rbf = RBFNet(x, y, 20, 500, mu, 0.15, save=False)
                rbf.train_lm_static()   # train the network with the static damping factor
                L.append(rbf.losses)

            # store to file for later use
            L = np.array([np.array(loss) for loss in L], dtype=object)
            with open('data/rbf_lm_mu_influence.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(L)
        else:
            # plotting code
            with open('data/rbf_lm_mu_influence.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float(r) for r in row]
                    L.append(row)
            net = pickle.load(open('data/rbf_lm_testing.pkl', 'rb'))
            L.append(net.losses[-1]['train_RSS'])
            plt.figure(figsize=(10, 6))
            for i, mu in enumerate(mus):
                plt.semilogy(L[i], label=rf'$\mu={mu}$')
            plt.semilogy(L[-1], label=r'dynamic $\mu$')
            plt.xlabel('Epoch')
            plt.ylabel('RSS')
            plt.title('Influence of damping factor on convergence (LMA)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('plots/rbf_lm_mu_influence.png', dpi=300)
            plt.show()

    @staticmethod
    def optimize_structure(x, y, plot=False):
        """Calculate and plot the influence of the hidden dimension on the accuracy of the network. The hidden dimension is varied and the
        final loss values are plotted to show the effect of the network structure on the convergence of the network."""
        hds = np.arange(1, 51)  # different hidden dimensions to test
        L = []
        if not plot:
            for hd in hds:  # iterate over the different hidden dimensions
                print(hd)
                rbf = RBFNet(x, y, hd, 500, 0.01, 0.15, save=True, input_dim=3, filename=f'data/test/rbfLM_{hd}.pkl')
                rbf.train_lm()  # train the network with the (dynamic μ) LMA
                L.append((rbf.losses[-1]['train_RSS'], rbf.losses[-1]['val_RSS']))

            # store to file for later use
            L = np.array([np.array(loss) for loss in L], dtype=object)
            with open('data/rbf_lm_hd_influence.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(L)
        else:
            # plotting code
            with open('data/rbf_lm_hd_influence.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float(r) for r in row]
                    L.append(row)

            L = np.array(L)
            plt.figure(figsize=(10, 6))
            plt.semilogy(hds, L[:, 0], label='Validation data')
            plt.semilogy(hds, L[:, 1], label='Special validation data')
            plt.xlabel('Epoch')
            plt.ylabel('RSS')
            plt.title('Influence of hidden dimension on convergence (LMA)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('plots/rbf_lm_hd_influence.png', dpi=300)
            plt.show()


if __name__ == '__main__':
    ols_data = np.loadtxt('data/output.csv', delimiter=',')  # load full reconstructed data
    Y, X = ols_data[:, 0], ols_data[:, 1:]
    Y, X = treat_data((Y, X))

    # rbf = RBFNet(X, Y, 30, 500, 0.01, 0.3, retry=True, filename='data/rbf_lm_testing.pkl', input_dim=3)
    # rbf.train_lm()

    net = pickle.load(open('data/rbf_lm_testing.pkl', 'rb'))
    plt.semilogy([net.losses[i]['train_RSS'] for i in range(len(net.losses))])
    plt.show()
    plt.semilogy(net.mus)
    plt.show()
    RBFNet.plot_lm('', net)

    # RBFNet.damping_effect(X, Y)
    # RBFNet.damping_effect(X, Y, plot=True)

    # RBFNet.optimize_structure(X, Y)
    # RBFNet.optimize_structure(X, Y, plot=True)
