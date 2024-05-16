import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def train_data(filename='data/F16traindata_CMabV_2024.csv'):
    """Read the data provided for the state estimation problem, stored as a .csv in the filename provided.
    This outputs the Cm measurements, system output vector Zk = [α, β, V] and system input vector Uk = [u̇, v̇, ẇ]."""
    df = pd.read_csv(filename, header=None)
    Cm = df.iloc[:, 0].to_numpy()
    Zk = df.iloc[:, 1:4].to_numpy()
    Uk = df.iloc[:, 4:7].to_numpy()
    return Cm, Zk, Uk


def validation_data(filename='data/F16validationdata_2024.csv'):
    """Read the special validation data provided for the prarmeter estimation problems, stored as a .csv in the filename provided.
    This outputs the Cm measurements on a grid of α and β values."""
    df = pd.read_csv(filename, header=None)
    Cm_val = df.iloc[:, 0].to_numpy()
    alpha_val = df.iloc[:, 1].to_numpy()
    beta_val = df.iloc[:, 2].to_numpy()
    return Cm_val, alpha_val, beta_val


def treat_data(data):
    """Treat parameter estimation dataset to remove outlier data. A datapoint is considered an outlier if Cm[i] > 0.2."""
    Y, X = data
    idx = np.setdiff1d(np.arange(len(Y)), np.where(Y > 0.2))
    Y = Y[idx]
    X = X[idx]
    return Y, X


def plot_Cm_glitch(Cm):
    """Show the outlier data in the Cm measurements."""
    colours = ['b' if x < 0.2 else 'r' for x in Cm]
    plt.scatter(np.arange(len(Cm)), Cm, s=0.4, c=colours)
    plt.xlabel('Sample')
    plt.ylabel(r'$C_m$ [-]')
    plt.grid()
    plt.title('F16 $C_m$ Glitch')
    plt.tight_layout()
    plt.savefig('plots/Cm_glitch.png')
    plt.show()
