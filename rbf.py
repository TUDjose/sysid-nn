import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read_data import treat_data
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay
torch.random.manual_seed(1)


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight)


class RBFLayer(nn.Module):
    def __init__(self, in_features, out_features, centers, labels):
        super(RBFLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))
        self.centers = centers
        self.labels = labels
    def forward(self, x):
        c = torch.tensor(self.centers, dtype=torch.float32)
        X = torch.zeros(x.shape[0], c.shape[0], x.shape[1])
        for i in range(x.shape[1]):
            xx = x[:, i].reshape(-1, 1)
            cc = c[:, i].reshape(-1, 1).t()
            X[:, :, i] = (xx - cc).pow(2)
        self.R = X
        y = self.weight[0, :]**2 * X[:, :, 0] + self.weight[1, :]**2 * X[:, :, 1] + self.weight[2, :]**2 * X[:, :, 2]
        return y

class RBFActivation(nn.Module):
    def __init__(self):
        super(RBFActivation, self).__init__()

    def forward(self, x, a=1.):
        return a * torch.exp(-x)


class RBFNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features, centers):
        super(RBFNet, self).__init__()
        self.layer1 = RBFLayer(in_features, hidden_dim, *centers)
        self.layer2 = Linear(hidden_dim, out_features)
        self.activation = RBFActivation()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x

    @staticmethod
    def get_centers(x, hidden_dim):
        x = x.detach().numpy()
        kmeans = KMeans(n_clusters=hidden_dim, random_state=0).fit(x)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        return centers, labels


def train_lin_regress(X, Y, hidden_dim):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=0)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    centers_train = RBFNet.get_centers(X_train_tensor, hidden_dim)
    model = RBFNet(3, hidden_dim, 1, centers_train)

    params = list(model.parameters())
    updates = [params[0], torch.linalg.pinv(model.activation(model.layer1(X_train_tensor))) @ y_train_tensor]

    for idx, p in enumerate(model.parameters()):
        p.data = updates[idx]

    test_data = scaler.transform(X)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(test_data_tensor)

    return predictions, test_data

def train_lm(X, Y, num_epochs, mu, hidden_dim):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    centers_train = RBFNet.get_centers(X_train_tensor, hidden_dim)
    model = RBFNet(3, hidden_dim, 1, centers_train)
    criterion = nn.MSELoss()

    def jacbobian():
        W1, W2 = list(model.parameters())
        h = model.activation(model.layer1(X_train_tensor))
        y = model(X_train_tensor)

        dedw2 = -1 * (y_train_tensor - y) * h

        dedw1a = -1 * (y_train_tensor - y) @ W2.T * (-h) * 2 * W1[0, :] * model.layer1.R[:, :, 0]
        dedw1b = -1 * (y_train_tensor - y) @ W2.T * (-h) * 2 * W1[1, :] * model.layer1.R[:, :, 1]
        dedw1V = -1 * (y_train_tensor - y) @ W2.T * (-h) * 2 * W1[2, :] * model.layer1.R[:, :, 2]

        return torch.hstack([dedw1a, dedw1b, dedw1V]), dedw2

    losses = []
    for epoch in range(num_epochs):
        model.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        losses.append(loss.item())

        E = 0.5 * (y_train_tensor - outputs).pow(2)
        Jacobian = jacbobian()
        deltas = []
        for idx, J in enumerate(Jacobian):
            deltaW = torch.linalg.inv(J.t() @ J + mu * torch.eye(J.shape[1])) @ J.t() @ E
            deltaW = deltaW
            deltas.append(deltaW)

        for i, p in enumerate(model.parameters()):
            p.data -= deltas[i].reshape(p.shape)

        # Validation
        # model.eval()
        # with torch.no_grad():
        #     val_outputs = model(X_val_tensor)
        #     val_loss = criterion(val_outputs, y_val_tensor)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: ')

    test_data = scaler.transform(X)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(test_data_tensor)

    return predictions, losses, test_data


ols_data = np.loadtxt('data/output.csv', delimiter=',')  # load full reconstructed data
Y, X = ols_data[:, 0], ols_data[:, 1:]
Y, X = treat_data((Y, X))

# predictions, test_data = train_lin_regress(X, Y, 50)
#
# plt.figure(figsize=(12, 3))
# plt.plot(Y, label='True')
# plt.plot(predictions, label='Predictions')
# plt.legend()
# plt.show()

predictions, losses, test_data = train_lm(X, Y, 2000, 0.002, 15)

tri = Delaunay(test_data[:, :2])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_data[:, 0], test_data[:, 1], Y, c='r', lw=1.3)
ax.plot(test_data[:, 0], test_data[:, 1], predictions.reshape(-1), c='b', lw=1.3)
# ax.plot_trisurf(test_data[:, 0], test_data[:, 1], predictions.reshape(-1), triangles=tri.simplices)
ax.set_zlim(-0.12, 0)
plt.show()

plt.semilogy(losses)
plt.show()

plt.figure(figsize=(12, 3))
plt.plot(Y, label='True')
plt.plot(predictions, label='Predictions')
plt.legend()
plt.show()
