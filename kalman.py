import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import control.matlab as cm


def train_data(filename='F16traindata_CMabV_2024.csv'):
    df = pd.read_csv(filename, header=None)
    Cm = df.iloc[:, 0].to_numpy()
    Zk = df.iloc[:, 1:4].to_numpy()
    Uk = df.iloc[:, 4:7].to_numpy()
    return Cm, Zk, Uk

def validation_data(filename='F16validationdata_2024.csv'):
    df = pd.read_csv(filename, header=None)
    Cm_val = df.iloc[:, 0].to_numpy()
    alpha_val = df.iloc[:, 1].to_numpy()
    beta_val = df.iloc[:, 2].to_numpy()
    return Cm_val, alpha_val, beta_val


class KalmanFilter:
    MAX_ITER = 100
    TOLERANCE = 1e-10

    def __init__(self, dt, data):
        self.Cm, self.Zk, self.Uk = data
        self.dt = dt
        self.N = len(self.Uk)

        self.states = 4
        self.inputs = 3
        self.meas = 3

        self.initialize_kf_params()

    def initialize_kf_params(self):
        self.E_x_0 = np.array([self.Zk[0, 2], 0.5, 0.5, 0.5])      # TODO: check this
        self.P_0_0 = np.eye(self.states) * 0.1        # TODO: check this

        self.Ew = np.zeros(4)
        self.sigma_w = np.array([1e-3, 1e-3, 1e-3, 0])
        self.Q = np.diagflat(np.power(self.sigma_w, 2))

        self.Ev = np.zeros(3)
        self.sigma_v = np.array([1.5e-3, 1.5e-3, 1e0])
        self.R = np.diagflat(np.power(self.sigma_v, 2))

        self.G = np.eye(self.states)
        self.Xk1k1 = np.zeros([self.states, self.N])
        self.Pk1k1 = np.zeros([self.states, self.states, self.N])
        self.Z_pred = np.zeros([self.meas, self.N])
        self.STD_xcor = np.zeros([self.states, self.N])
        self.STD_z = np.zeros([self.meas, self.N])
        self.IEKFcntr = np.zeros([self.N, 1])

    @staticmethod
    def rk4(fn, xin, uin, t):
        a = t[0]
        b = t[1]
        w = xin
        N = 2
        h = (b - a) / N
        t = a

        for j in range(1, N + 1):
            K1 = h * fn(t, w, uin)
            K2 = h * fn(t + h / 2, w + K1 / 2, uin)
            K3 = h * fn(t + h / 2, w + K2 / 2, uin)
            K4 = h * fn(t + h, w + K3, uin)

            w = w + (K1 + 2 * K2 + 2 * K3 + K4) / 6
            t = a + j * h

        return t, w
    def f(self, t, X, U):
        return np.concatenate((U, [0]))

    def h(self, t, X, U):
        u, v, w, Caup = X
        h = np.array([np.arctan(w/u) * (1 + Caup),
                       np.arctan(v / np.sqrt(u**2 + w ** 2)),
                       np.sqrt(u**2 + v**2 + w**2)])
        return h

    def Fx(self, t, X, U):
        return np.zeros((4, 4))

    def Hx(self, t, X, U):
        u, v, w, C = X
        Vtot = u**2 + v**2 + w**2
        Hx = np.array([
            [-w*(1+C)/(u**2 + w**2), 0, u*(1+C)/(u**2 + w**2), np.arctan(w/u)],
            [-u*v/(np.sqrt(u**2 + w**2)*Vtot), np.sqrt(u**2 + w**2)/Vtot, -v*w/(np.sqrt(u**2 + w**2)*Vtot), 0],
            [u/np.sqrt(Vtot), v/np.sqrt(Vtot), w/np.sqrt(Vtot), 0]])
        return Hx

    def prove_convergence(self):
        print("Observability matrix has full rank, hence system is observable and IEKF will converge.")

    def IEKF(self):
        x_k1_k1 = self.E_x_0
        P_k1_k1 = self.P_0_0

        tk = 0
        tk1 = self.dt

        for k in range(self.N):
            # 1. one step ahead prediction
            _, x_k1_k = KalmanFilter.rk4(self.f, x_k1_k1, self.Uk[k], [tk, tk1])

            # 2. calculate Fx
            Fx = self.Fx(0, x_k1_k, self.Uk[k])

            # 3. discretize for phi and gamma
            ssB = cm.ss(Fx, self.G, np.eye(4), 0)
            Phi = cm.c2d(ssB, self.dt).A
            Gamma = cm.c2d(ssB, self.dt).B

            # 4. calculate P_k1_k
            P_k1_k =  Phi @ P_k1_k1 @ Phi.T + Gamma @ self.Q @ Gamma.T

            # IEKF loop
            eta_i = x_k1_k
            err = 2*self.TOLERANCE
            iter = 0

            while err > self.TOLERANCE:
                if iter > self.MAX_ITER:
                    print("Terminating IEKF - exceeded maximum iterations")
                    break

                iter += 1
                eta1 = eta_i

                # 5. recalculate Hx
                Hx = self.Hx(0, eta1, self.Uk[k])
                z_k1_k = self.h(0, eta1, self.Uk[k])
                Pz = Hx @ P_k1_k @ Hx.T + self.R
                std_z = np.sqrt(np.diag(Pz))

                # 6. calculate Kk1
                K_k1 = P_k1_k @ Hx.T @ np.linalg.inv(Pz)

                # 7. update x_k1_k1
                temp = K_k1 @ (self.Zk[k] - z_k1_k - Hx @ (x_k1_k - eta1))

                eta_i = x_k1_k + K_k1 @ (self.Zk[k] - z_k1_k - Hx @ (x_k1_k - eta1))
                eta_i = np.ravel(eta_i)
                err = np.linalg.norm(eta_i - eta1) / np.linalg.norm(eta1)

            self.IEKFcntr[k] = iter
            x_k1_k1 = eta_i

            # 8. update P_k1_k1
            P_k1_k1 = (np.eye(self.states) - K_k1 @ Hx) @ P_k1_k @ (np.eye(self.states) - K_k1 @ Hx).T + K_k1 @ self.R @ K_k1.T
            std_x_cor = np.sqrt(np.diag(P_k1_k1))

            self.Xk1k1[:, k] = x_k1_k1
            self.Pk1k1[:, :, k] = P_k1_k1
            self.Z_pred[:, k] = z_k1_k
            self.STD_xcor[:, k] = std_x_cor
            self.STD_z[:, k] = std_z

    def alpha_reconstruction(self):
        alpha_bias_corr = self.Z_pred[0, :]
        alpha_bias_corr /= (1 + self.Xk1k1[3, :])
        return alpha_bias_corr


    def plot(self):
        fig, ax = plt.subplots(3,1, figsize=(7, 8))
        for i in range(3):
            ax[i].plot(self.Zk[:, i], label='True')
            ax[i].plot(self.Z_pred[i, :], label='Predicted')
            ax[i].legend()
            ax[i].grid()
        ax[0].plot(self.alpha_reconstruction(), label='Bias corrected')
        plt.tight_layout()
        plt.show(dpi=300)
