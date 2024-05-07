import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from read_data import treat_data
import matplotlib.pyplot as plt



class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class FeedForwardNet(nn.Module):
    def __init__(self, in_features, hidden_dim, out_features):
        super(FeedForwardNet, self).__init__()
        self.layer1 = Linear(in_features, hidden_dim)
        self.layer2 = Linear(hidden_dim, out_features)
        self.activation = torch.tanh

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.layer2(x)
        return x


def train(X, Y, num_epochs, lr, hidden_dim):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    model = FeedForwardNet(3, hidden_dim, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses = []
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(X_train_tensor)

        # Compute loss
        loss = criterion(outputs, y_train_tensor)
        losses.append(loss.item())

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)

        # Print progress
        if (epoch + 1) % 100 == 0:
            print(model.layer1.bias.detach().numpy())
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

    test_data = scaler.transform(X)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(test_data_tensor)

    return predictions, losses, test_data


def train_lm(X, Y, num_epochs, mu, hidden_dim):
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, requires_grad=True)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, requires_grad=True)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    model = FeedForwardNet(3, hidden_dim, 1)
    criterion = nn.MSELoss()

    def jacbobian():
        W1, b1, W2, b2 = list(model.parameters())
        h = model.activation(model.layer1(X_train_tensor))
        y = model(X_train_tensor)

        dedw2 = -1 * (y_train_tensor - y) * h
        dedb2 = -1 * (y_train_tensor - y)

        dedw1a = -1 * (y_train_tensor - y) @ W2 * (1 - h ** 2) * X_train_tensor[:, 0].reshape(-1, 1)
        dedw1b = -1 * (y_train_tensor - y) @ W2 * (1 - h ** 2) * X_train_tensor[:, 1].reshape(-1, 1)
        dedw1V = -1 * (y_train_tensor - y) @ W2 * (1 - h ** 2) * X_train_tensor[:, 2].reshape(-1, 1)
        dedb1 = -1 * (y_train_tensor - y) @ W2 * (1 - h ** 2)

        return torch.hstack([dedw1a, dedw1b, dedw1V]), dedb1, dedw2, dedb2

    val_loss_prev = 0.
    losses = []
    for iteration in range(num_epochs):
        try:
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

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)

            if (iteration + 1) % 100 == 0:
                print(f'Epoch [{iteration + 1}/{num_epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}, mu: {mu:.6f}')
        except KeyboardInterrupt:
            break


    test_data = scaler.transform(X)
    test_data_tensor = torch.tensor(test_data, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        predictions = model(test_data_tensor)

    return predictions, losses,  test_data


def lr_effect_on_convergence(X, Y, type='backprop'):
    LT = []
    lrs = [1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.002, 0.003, 0.005, 0.01, 0.1]
    for lr in lrs:
        if type == 'backprop':
            predictions, losses, test_data = train(X, Y, 3000, lr, 15)
        elif type == 'lm':
            predictions, losses, test_data = train_lm(X, Y, 3000, lr, 15)
        LT.append(losses)

    for loss in LT:
        plt.semilogy(loss, label=f'lr = {lrs[LT.index(loss)]}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('plots/lr_effect_on_convergence.png', dpi=300)
    plt.show()

def neuron_effect(X, Y, type='backprop'):
    LT = []
    neurons = [1, 2, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for neuron in neurons:
        if type == 'backprop':
            predictions, losses, test_data = train(X, Y, 3000, 0.01, neuron)
        elif type == 'lm':
            predictions, losses, test_data = train_lm(X, Y, 3000, 0.01, neuron)
        LT.append(losses)

    for loss in LT:
        plt.semilogy(loss, label=f'neurons = {neurons[LT.index(loss)]}')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig('plots/neuron_effect_on_convergence.png', dpi=300)
    plt.show()



ols_data = np.loadtxt('data/output.csv', delimiter=',')  # load full reconstructed data
Y, X = ols_data[:, 0], ols_data[:, 1:]
Y, X = treat_data((Y, X))

predictions, losses,  test_data = train(X, Y, 10000, 0.01, 15)

plt.figure(figsize=(12, 3))
plt.plot(Y, label='True')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.show()

plt.semilogy(losses)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(test_data[:, 0], test_data[:, 1], predictions.detach().numpy().ravel(), c='r', lw=1.3, label='FNN')
ax.plot(test_data[:, 0], test_data[:, 1], Y.ravel(), c='k', lw=1.3, label='Measured', alpha=0.5)
ax.set_xlabel(r'$\alpha$ [rad]')
ax.set_ylabel(r'$\beta$ [rad]')
ax.set_zlabel(r'$C_m$ [-]')
ax.set_zlim(-0.2, 0)
plt.legend()
plt.tight_layout()
plt.savefig('plots/fnn.png', dpi=300)
plt.show()

# lr_effect_on_convergence(X, Y)
# neuron_effect(X, Y)
