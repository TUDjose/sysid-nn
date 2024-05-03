import pandas as pd
import numpy as np

def train_data(filename='data/F16traindata_CMabV_2024.csv'):
    df = pd.read_csv(filename, header=None)
    Cm = df.iloc[:, 0].to_numpy()
    Zk = df.iloc[:, 1:4].to_numpy()
    Uk = df.iloc[:, 4:7].to_numpy()
    return Cm, Zk, Uk

def validation_data(filename='data/F16validationdata_2024.csv'):
    df = pd.read_csv(filename, header=None)
    Cm_val = df.iloc[:, 0].to_numpy()
    alpha_val = df.iloc[:, 1].to_numpy()
    beta_val = df.iloc[:, 2].to_numpy()
    return Cm_val, alpha_val, beta_val

def treat_data(data):
    Y, X = data
    idx = np.setdiff1d(np.arange(len(Y)), np.where(Y > 0.2))
    Y = Y[idx]
    X = X[idx]
    return Y, X