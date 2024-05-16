import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read_data import treat_data, validation_data
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import Delaunay
import csv
from tqdm import tqdm


class Linear(nn.Module):
    """Linear layer of the FNN, containing the weights and biases of the layer."""
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))  # layer weights
        self.bias = nn.Parameter(torch.randn(out_features))     # layer biases

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias     # calculate the forward pass of the linear layer (y = Wx + b)


class Net(nn.Module):
    """Feedforward Neural Network architecture, with two linear layers and a tanh activation function."""
    def __init__(self, in_features, hidden_dim, out_features):
        super(Net, self).__init__()
        self.layer1 = Linear(in_features, hidden_dim)   # first layer (input -> hidden)
        self.layer2 = Linear(hidden_dim, out_features)  # second layer (hidden -> output)
        self.activation = torch.tanh                    # activation function

    def forward(self, x):
        """Calculate the forward pass of the network, by passing the output of the first linear layer into the tanh activation function,
        then pass that through the second linear layer."""
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


class FNN:
    """Class to hold and train the FNN through backpropagation/gradient descent and Levenberg-Marquardt algorithm. Class also contains
    methods for network visualization, analysis and validation.
    :param X: input data
    :param Y: output data
    :param hidden_dim: number of neurons in the hidden layer
    :param epochs: number of epochs to train the network (LMA)
    :param mu: damping factor for LMA or learning rate for Adam
    :param goal: goal value for the loss function
    :param optimizer: optimization algorithm to use (Adam or LM)
    :param save: whether to save the trained model
    :param filename: filename to save the model to
    :param retry: whether to retry training if the model fails to converge (LMA only)
    """
    def __init__(self, X, Y, hidden_dim, epochs, mu, goal, optimizer, save=True, filename=None, retry=False, input_dim=3):
        self.X, self.Y = X, Y
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.epochs = epochs
        self.mu = mu
        self.goal = goal
        self.optimizer = optimizer
        self.filename = f'data/fnn_{optimizer}_{hidden_dim}.pkl' if not filename else filename
        self.retry = retry
        self.save = save

        self.init_data(X, Y)
        self.model = Net(input_dim, hidden_dim, 1)      # initialize the FNN model

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
        """Calculate the mean squared error (MSE) for the different datasets."""
        MSE = nn.MSELoss()  # mean squared error
        RSS = lambda x, y: (x - y).pow(2).sum()  # residual sum of squares
        losses = {
            'train': MSE(out['train'], self.y_train_tensor),
            'val': MSE(out['val'], self.y_val_tensor),
            'test': MSE(out['test'], self.y_test_tensor),
            'special': MSE(out['special'], self.y_spec_tensor) if self.input_dim == 2 else None,
            'train_RSS': RSS(out['train'], self.y_train_tensor),
            'val_RSS': RSS(out['val'], self.y_val_tensor),
            'test_RSS': RSS(out['test'], self.y_test_tensor),
            'special_RSS': RSS(out['special'], self.y_spec_tensor) if self.input_dim == 2 else None
        }
        return losses

    def train_adam(self, decay=True):
        """Train the FNN using gradient descent and the Adam optimization algorithm."""
        #  define the training optimizer as Adam using Pytorch's implementation
        optimizer = optim.Adam(self.model.parameters(), lr=self.mu)
        self.losses = [self.get_losses(self.get_outputs())]
        epoch = 0
        pbar = tqdm(desc='Epochs', unit=' epoch')
        try:
            while self.losses[-1]['train'] > self.goal:     # train until the goal value is reached
                self.model.train()      # set the model to training mode
                outputs = self.get_outputs()    # calculate the outputs of the network
                loss = self.get_losses(outputs)     # calculate the loss of the network
                self.losses.append(loss)

                optimizer.zero_grad()       # zero the gradients
                loss['train'].backward()    # backpropagate the loss (i.e. calculate the gradient of the loss/error w.r.t. the weights)
                optimizer.step()        # update the weights using the optimizer

                epoch += 1
                pbar.update(1)

                if epoch > 25000 and self.losses[-1]['train'] < 2e-5:   # stop training if model takes too long to converge
                    break

                if decay:       # apply a decay to the learning rate to improve stability of convergence
                    if self.losses[-1]['train'] <= 4e-5:
                        optimizer.param_groups[0]['lr'] = 0.001
                    elif self.losses[-1]['train'] <= 2e-5:
                        optimizer.param_groups[0]['lr'] = 0.0001

        except KeyboardInterrupt:
            pass

        pbar.close()
        self.model.eval()
        with torch.no_grad():
            self.predictions = self.model(self.X_val_tensor)
            self.outputs = self.get_outputs()

        print(f'Final Loss: {self.losses[-1]["train"]} at epoch {epoch}')

        if self.save:
            pickle.dump(self, open(self.filename, 'wb'))

    def train_lm(self):
        """Train the FNN using the Levenberg-Marquardt algorithm. The different LMA steps are detailed in this method"""
        def jacobian():
            """Calculate the Jacobian matrix components of the error vector w.r.t. the weights vector"""
            W1, b1, W2, b2 = list(self.model.parameters())      # learnable parameters of the network
            h = self.model.activation(self.model.layer1(self.X_train_tensor)) # output of tanh activation function
            y = self.model(self.X_train_tensor)     # network output

            # derivatives of the error w.r.t. the weights and biases of the second layer
            dedw2 = -1 * (self.y_train_tensor - y) * h
            dedb2 = -1 * (self.y_train_tensor - y)
            # derivatives of the error w.r.t. the weights and biases of the first layer for inputs α and β
            dedw1a = -1 * (self.y_train_tensor - y) @ W2 * (1 - h ** 2) * self.X_train_tensor[:, 0].reshape(-1, 1)
            dedw1b = -1 * (self.y_train_tensor - y) @ W2 * (1 - h ** 2) * self.X_train_tensor[:, 1].reshape(-1, 1)
            dedb1 = -1 * (self.y_train_tensor - y) @ W2 * (1 - h ** 2)

            if self.input_dim == 3:
                # derivatives of the error w.r.t. the weights of the first layer for input V
                dedw1V = -1 * (self.y_train_tensor - y) @ W2 * (1 - h ** 2) * self.X_train_tensor[:, 2].reshape(-1, 1)
                return torch.hstack([dedw1a, dedw1b, dedw1V]), dedb1, dedw2, dedb2
            else:
                return torch.hstack([dedw1a, dedw1b]), dedb1, dedw2, dedb2

        pbar = tqdm(desc='Epochs', unit=' epoch')
        update = True
        mu_inc, mu_dec = 10, 10   # increase and decrease factors for damping factor within the LMA
        self.losses = []

        for epoch in range(self.epochs):    # iterate over the maximum number of epochs
            try:
                """1. If network should be updated, calculate the outputs and losses"""
                if update:
                    self.model.zero_grad()   # zero the gradients of the model
                    outputs = self.get_outputs()    # calculate the outputs of the network
                    loss = self.get_losses(outputs)    # calculate the loss of the network
                    self.losses.append(loss)
                    E = 0.5 * (self.y_train_tensor - outputs['train']).pow(2)   # calculate the error vector
                    Jacobian = jacobian()   # calculate the Jacobian matrix

                """2. Calculate the LMA weight updates"""
                deltas = []
                for idx, J in enumerate(Jacobian):
                    deltaW = torch.linalg.inv(J.t() @ J + self.mu * torch.eye(J.shape[1])) @ J.t() @ E   # calculate the LMA weight updates
                    deltas.append(deltaW)

                """3. Update the network parameters  on a dummy network and calculate its loss"""
                # create dummy network and make it equal to the current network
                MODEL = FNN(self.X, self.Y, self.hidden_dim, 1, self.mu, self.goal, self.optimizer, input_dim=self.input_dim)
                MODEL.model.load_state_dict(self.model.state_dict())
                for i, p in enumerate(MODEL.model.parameters()):
                    p.data -= deltas[i].reshape(p.shape)    # update the parameters of the dummy network
                loss_new = MODEL.get_losses(MODEL.get_outputs())    # calculate the loss of the dummy network

                """4. If the dummy loss is lower than the current loss, decrease μ and update the current network parameters
                      If the dummy loss is higher than the current loss, increase μ and do not update the current network parameters"""
                if loss_new['train_RSS'] < self.losses[-1]['train_RSS']:
                    self.mu /= mu_dec   # decrease damping factor
                    update = False
                    for i, p in enumerate(self.model.parameters()):
                        p.data -= deltas[i].reshape(p.shape)    # update the parameters of the current network
                    self.losses.append(self.get_losses(self.get_outputs()))
                else:
                    self.mu *= mu_inc   # increase damping factor
                    update = True
                    # do not update params

                if self.mu >= 10e8:     # stop training if damping factor is too high
                    self.epochs = epoch
                    break

                pbar.update(1)

            except torch._C._LinAlgError as e:  # if the Jacobian matrix ends up being singular, restart the training automatically
                print(f'LinAlgError: {e}')
                FNN(self.X, self.Y, self.hidden_dim, 1000, 0.01, self.goal, self.optimizer, retry=True,
                          input_dim=self.input_dim, filename=self.filename).train_lm()
                break

            except KeyboardInterrupt:
                break

        pbar.close()
        #  if the goal is not reached and retry is enabled, restart the training automatically
        if self.losses[-1]['train_RSS'] > self.goal and self.retry:
            print(f'Failed to converge to goal, loss={self.losses[-1]["train_RSS"]:.6f}, trying again ...')
            FNN(self.X, self.Y, self.hidden_dim, 1000, 0.01, self.goal, self.optimizer, retry=True,
                      input_dim=self.input_dim, filename=self.filename).train_lm()

        else:
            # if the goal is reached, evaluate the network on the validation data
            self.model.eval()
            with torch.no_grad():
                self.predictions = self.model(self.X_val_tensor)
                self.outputs = self.get_outputs()   # calculate the outputs of the network
            print(f'Final Loss (RSS/MSE): {self.losses[-1]["train_RSS"]:.6f}/{self.losses[-1]["train"]:.6f} at epoch {epoch}')

            if self.save:
                pickle.dump(self, open(self.filename, 'wb'))    # save the trained model to file for later use


    @staticmethod
    def plot_fnn(filename, net=None, special=False):
        """Plot the FNN predictions on the input data and the actual measurements."""
        if not net:
            fnn = pickle.load(open(filename, 'rb'))   # load the FNN model
            x = fnn.X_val_tensor.detach().numpy()   # get the input data
            y = fnn.y_val_tensor.detach().numpy()   # get the output data
            p = fnn.predictions.detach().numpy()    # get the predictions
        else:
            fnn = net
            x = fnn.X_val_tensor.detach().numpy()
            y = fnn.y_val_tensor.detach().numpy()
            p = fnn.predictions.detach().numpy()
            if special:
                x = fnn.X_spec_tensor.detach().numpy()
                y = fnn.y_spec_tensor.detach().numpy()
                p = fnn.special_val.detach().numpy()

        tri = Delaunay(x[:, :2])    # Delaunay triangulation of the predictions to show a surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=33, azim=70)
        ax.scatter(x[:, 0], x[:, 1], y.reshape(-1), c='k', s=0.3, label='Measurements')
        ax.plot_trisurf(x[:, 0], x[:, 1], p.reshape(-1), triangles=tri.simplices, cmap='coolwarm', label='FNN')
        ax.set_zlim(-0.12, 0)
        ax.set_title(rf'$C_m(\alpha, \beta)$ - FNN on input data ({fnn.optimizer})', y=0.98)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.set_zlabel(r'$C_m$')
        ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
        plt.tight_layout()
        plt.savefig(f'plots/fnn_{fnn.optimizer}.png', dpi=300)
        plt.show()

    @staticmethod
    def compare_lm_adam():
        """Compare the MSE convergence of the FNN trained with Adam and the FNN trained with the Levenberg-Marquardt algorithm."""
        lm = pickle.load(open('data/fnn_LM_30.pkl', 'rb'))    # load the FNN trained with LM
        adam = pickle.load(open('data/test/adam_30', 'rb'))  # load the FNN trained with Adam
        # plot validation loss curves for both models
        plt.loglog([lm.losses[i]['val'].item() for i in range(len(lm.losses))], label='LMA')
        plt.loglog(adam.losses, label='Adam')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('FNN MSEs for different optimization algorithms')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig('plots/fnn_losses_adam_lm.png', dpi=300)
        plt.show()

    @staticmethod
    def check_init_conditions(x, y, optimizer, plot=False):
        """Check the influence of different initial conditions on the convergence of the FNN. The same network is trained 10 times and
        the different results are plotted to show the differences between them."""
        l = []
        if not plot:
            for i in range(10):    # train the network 10 times with different initial conditions
                print(i)
                if optimizer == 'adam':    # train the network with Adam and store the losses
                    fnn = FNN(x, y, 10, ..., 0.1, 2e-5, 'adam', save=False)
                    fnn.train_adam()
                    l.append([fnn.losses[i]['val'] for i in range(len(fnn.losses))])
                elif optimizer == 'LM':   # train the network with LMA and store the losses
                    fnn = FNN(x, y, 30, 5000, 0.01, 0.2, 'LM', save=False, retry=True)
                    fnn.train_lm()
                    l.append([fnn.losses[i]['val'] for i in range(len(fnn.losses))])
            l = np.array([np.array(loss) for loss in l], dtype=object)

            with open(f'data/{optimizer}_init_cond.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(l)

        else:
            # read data from file and plot the losses
            with open(f'data/{optimizer}_init_cond.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float(r) for r in row]
                    l.append(row)

            for i, loss in enumerate(l):
                plt.semilogy(loss)
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title(f'FNN MSEs for different initializations ({optimizer})')
            plt.grid()
            plt.tight_layout()
            plt.savefig(f'plots/{optimizer}_init_cond.png', dpi=300)
            plt.show()

    @staticmethod
    def lr_influence(x, y, plot=False):
        """Calculate the influence of the learning rate on the convergence of the FNN using the Adam optimization algorithm. The FNN is
        trained with different static learning rates and the MSEs are plotted to show the differences in convergence between them."""
        lrs = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]  # different learning rates to test
        L = []
        if not plot:
            for lr in lrs:  # iterate over the different learning rates
                print(lr)
                fnn = FNN(x, y, 7, ..., lr, 1.8e-5, 'adam', save=False)
                fnn.train_adam(decay=False)   # train the network with the given learning rate
                L.append([fnn.losses[i]['val'] for i in range(len(fnn.losses))])
            L = np.array([np.array(loss) for loss in L], dtype=object)
            with open(f'data/adam_lr_influence.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(L)

        else:
            # read data from file and plot the losses
            with open(f'data/adam_lr_influence.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float(r) for r in row]
                    L.append(row)

            plt.figure(figsize=(10, 6))
            for i, loss in enumerate(L):
                if i < 3:
                    plt.semilogy(loss, label=f'lr={lrs[i]}', lw=2)
                else:
                    plt.semilogy(loss, label=f'lr={lrs[i]}', lw=1)

            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.title('Influence of learning rate on convergence (Adam)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('plots/adam_lr_influence.png', dpi=300)
            plt.show()

    @staticmethod
    def hd_influence(x, y, plot=False, best=None):
        """Calculate and plot the influence of number of hidden neurons on the accuracy of the FNN using the Adam optimization algorithm
        and tradeoff between model accuracy and complexity."""
        hds = np.arange(1, 31)  # different hidden dimensions to test
        L = []
        if not plot:
            for hd in hds:  # iterate over the different hidden dimensions
                print(hd)
                fnn = FNN(x, y, hd, ..., 0.1, 2e-5, 'adam', save=True, filename=f'data/test/adam_{hd}')
                fnn.train_adam()    # train the network with the given hidden dimension
                L.append((fnn.losses[-1]['train'], fnn.losses[-1]['special']))

            L = np.array([np.array(loss) for loss in L], dtype=object)
            with open(f'data/adam_hd_influence.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(L)
        else:
            # read data from file and plot the losses and the best structure (in this case 12 hidden neurons, done manually)
            with open(f'data/adam_hd_influence.csv', 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    row = [float(r) for r in row]
                    L.append(row)

            L = np.array(L)
            plt.figure(figsize=(8, 5))
            plt.semilogy(hds, L[:, 0], label='Validation data')
            plt.semilogy(hds, L[:, 1], label='Special validation data')
            plt.vlines(12, 1e-5, 1e-2, colors='r', linestyles='dashed', label='Best structure')
            plt.xlabel('Hidden Dimension')
            plt.ylabel('MSE')
            plt.title('Influence of hidden dimension on convergence (Adam)')
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig('plots/adam_hd_influence.png', dpi=300)
            plt.show()

    @staticmethod
    def plot_special():
        """Plot the FNN predictions on the special validation data and the actual measurements and show the residual RMS."""
        net = pickle.load(open('data/test/adam_12', 'rb'))  # load the 'best' FNN model
        x2 = net.X_spec_tensor.detach().numpy()
        y2 = net.y_spec_tensor.detach().numpy()
        p2 = net.special_val.detach().numpy()

        rms = np.sqrt(np.mean((y2 - p2) ** 2))  # calculate the residual RMS
        print(f"The residual RMS of the FNN model of hidden dimension {net.hidden_dim} is: {rms:.4f}")

        # plot the FNN predictions on the special validation data
        tri2 = Delaunay(x2[:, :2])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=33, azim=70)
        ax.scatter(x2[:, 0], x2[:, 1], y2.reshape(-1), c='k', s=0.3, label='Measurements')
        ax.plot_trisurf(x2[:, 0], x2[:, 1], p2.reshape(-1), triangles=tri2.simplices, cmap='coolwarm', label='FNN')
        ax.set_zlim(-0.12, 0)
        ax.set_title(rf'$C_m(\alpha, \beta)$ - FNN on special validation data ({net.optimizer})', y=0.98)
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel(r'$\beta$')
        ax.set_zlabel(r'$C_m$')
        ax.legend(loc='upper right', bbox_to_anchor=(1.03, 0.90))
        plt.tight_layout()
        plt.savefig(f'plots/{net.optimizer}_spec_val_comp.png', dpi=300)
        plt.show()



if __name__ == '__main__':
    # ols_data = np.loadtxt('data/output.csv', delimiter=',')  # load full reconstructed data
    # Y, X = ols_data[:, 0], ols_data[:, 1:3]
    # Y, X = treat_data((Y, X))

    # fnn = FNN(X, Y, 30, 1000, 0.01, 0.18, optimizer='LM', retry=True, filename='data/fnn_LM_testing.pkl', input_dim=3)
    # fnn.train_lm()
    # fnn.plot_fnn('data/fnn_LM_testing.pkl')

    # FNN.check_init_conditions(X, Y, 'LM', plot=False)
    # FNN.check_init_conditions(X, Y, 'LM', plot=True)

    # fnn2 = FNN(X, Y, 12, ..., 0.1, 1.5e-5, optimizer='adam', input_dim=2)
    # fnn2.train_adam()
    # fnn2.plot_fnn('', fnn2)

    # FNN.check_init_conditions(X, Y, 'adam', plot=False)
    # FNN.check_init_conditions(X, Y, 'adam', plot=True)

    # FNN.compare_adam_lm('data/fnn_LM_30.pkl', 'data/fnn_adam_30.pkl')

    # FNN.lr_influence(X, Y, plot=False)
    # FNN.lr_influence(X, Y, plot=True)

    # FNN.hd_influence(X, Y, plot=True)

    net = pickle.load(open('data/test/adam_30', 'rb'))
    print(net.__dict__.keys())


