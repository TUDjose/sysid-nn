import numpy as np
import matplotlib.pyplot as plt
import torch


class RBFNet:
    def __init__(self, input_size, hidden_dim, output_size, R_input, epochs, training, mus, goal, min_grad, activation):
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.output_size = output_size

        self.N_weights = input_size * hidden_dim + hidden_dim * output_size
        self.weights1 = torch.randn(input_size, hidden_dim)
        self.weights_2 = torch.randn(output_size, hidden_dim)

        self.get_centers()
        self.R_input = R_input

        self.training = training
        self.epochs = epochs
        self.mu = mus[0]
        self.mu_inc = mus[1]
        self.mu_dec = mus[2]
        self.mu_max = mus[2]
        self.goal = goal
        self.min_grad = min_grad
        self.activation = activation

        self.results = []

    def get_centers(self):
        self.centers = None

    def train(self, X, Y):
        pass





in_features, hidden_dim, out_featurs = 3, 10, 1
R_input = 1
