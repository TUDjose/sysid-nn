import numpy as np
import matplotlib.pyplot as plt
import control.matlab as cm
import sympy
from tqdm import tqdm
plt.rcParams['text.usetex'] = True


class KalmanFilter:
    """
    This class contains all the necessary methods to perform an Iterative Extended Kalman Filter (IEKF) on the provided measurement data.
        :param data: tuple containing the data (Cm, Zk, Uk)
        :param n_states: number of states in system
    """

    def __init__(self, data, n_states):
        self.Cm, self.Zk, self.Uk = data
        self.dt = 0.01  # fixed time step, as per assignment details
        self.N = len(self.Uk)

        self.states = n_states
        self.inputs = self.Uk.shape[1]
        self.meas = self.Zk.shape[1]

        self.MAX_ITER = 100     # maximum number of iterations for IEKF
        self.EPSILON = 1e-10    # convergence criterion for iteration loop

        self.initialize_kf_params()

    def initialize_kf_params(self):
        self.E_x_0 = np.array([self.Zk[0, 2], 0.1, 0.1, 0.3])       # initial state estimate
        self.P_0_0 = np.eye(self.states)      # initial state prediction covariance matrix

        # system noise parameters
        self.sigma_w = np.array([1e-3, 1e-3, 1e-3, 0])
        self.Q = np.diagflat(np.power(self.sigma_w, 2))

        # sensor noise parameters
        self.sigma_v = np.array([1.5e-3, 1.5e-3, 1e0])
        self.R = np.diagflat(np.power(self.sigma_v, 2))

        self.G = np.eye(self.states)        # system noise matrix
        self.Xk1k1 = np.zeros([self.states, self.N])    # store state estimates
        self.Pk1k1 = np.zeros([self.states, self.states, self.N])   # store state estimation error covariance
        self.Z_pred = np.zeros([self.meas, self.N])     # store predicted measurement


    @staticmethod
    def rk4(fn, xin, uin, t):
        """Runge-Kutta method for solving one-step-ahead prediction of state estimate"""
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
        """(Trivial) System equations: xdot = f(x, u, t) : P1.2"""
        return np.concatenate((U, [0]))

    def h(self, t, X, U):
        """(Non-trivial) Output equations: z = h(x, u, t) : P1.2"""
        u, v, w, Caup = X
        h = np.array([np.arctan(w/u) * (1 + Caup),
                       np.arctan(v / np.sqrt(u**2 + w ** 2)),
                       np.sqrt(u**2 + v**2 + w**2)])
        return h

    def Fx(self, t, X, U):
        """Jacobian matrix of f(x, u, t) w.r.t. x"""
        return np.zeros((4, 4))

    def Hx(self, t, X, U):
        """Jacobian matrix of h(x, u, t) w.r.t. x"""
        u, v, w, Caup = X
        Vtot = u**2 + v**2 + w**2
        Hx = np.array([
            [-w*(1+Caup)/(u**2 + w**2), 0, u*(1+Caup)/(u**2 + w**2), np.arctan(w/u)],
            [-u*v/(np.sqrt(u**2 + w**2)*Vtot), np.sqrt(u**2 + w**2)/Vtot, -v*w/(np.sqrt(u**2 + w**2)*Vtot), 0],
            [u/np.sqrt(Vtot), v/np.sqrt(Vtot), w/np.sqrt(Vtot), 0]])
        return Hx

    def prove_convergence(self):
        """prove convergence of the kalman filtering. This is done by calculating the observability matrix and checking that its rank is
        equal to the number of states using a symbolic solver: P1.4"""
        u = sympy.symbols('u')
        v = sympy.symbols('v')
        w = sympy.symbols('w')
        Caup = sympy.symbols('Caup')

        udot = sympy.symbols('udot')
        vdot = sympy.symbols('vdot')
        wdot = sympy.symbols('wdot')

        x = sympy.Matrix([u, v, w, Caup])
        f = sympy.Matrix([udot, vdot, wdot, 0])
        h = sympy.Matrix([sympy.atan(w/u) * (1 + Caup), sympy.atan(v / sympy.sqrt(u**2 + w ** 2)), sympy.sqrt(u**2 + v**2 + w**2)])

        # calculate observability matrix from the Lie derivatives of the system and output equations
        O = sympy.Matrix()
        Hx = h.jacobian(x)
        O = O.col_join(Hx)
        Lfh = Hx @ f
        O = O.col_join(Lfh.jacobian(x))
        L2fh = Lfh.jacobian(x) @ f
        O = O.col_join(L2fh.jacobian(x))
        Onums = O.subs({'u': 100, 'v': 10, 'w': 10, 'Caup': 1})

        if Onums.rank() == self.states:
            print("Observability matrix has full rank, hence system is observable and IEKF will converge.")
        else:
            print("Observability matrix does not have full rank, hence system is not observable and IEKF will not converge.")

    def IEKF(self):
        """Implementation of the Iterative Extended Kalman Filter : P1.4"""
        x_k1_k1 = self.E_x_0    # set initial state estimate
        P_k1_k1 = self.P_0_0    # set initial state estimate covariance

        tk = 0
        tk1 = self.dt
        with tqdm(total=self.N) as pbar:
            for k in range(self.N):
                # 1. one step ahead prediction
                _, x_k1_k = KalmanFilter.rk4(self.f, x_k1_k1, self.Uk[k], [tk, tk1])

                # 2. calculate Fx
                Fx = self.Fx(0, x_k1_k, self.Uk[k])

                # 3. discretize system to find phi and gamma matrices
                ssB = cm.ss(Fx, self.G, np.eye(4), 0)
                Phi = cm.c2d(ssB, self.dt).A
                Gamma = cm.c2d(ssB, self.dt).B

                # 4. calculate state prediction error covariance matrix P_k1_k
                P_k1_k = Phi @ P_k1_k1 @ Phi.T + Gamma @ self.Q @ Gamma.T

                # IEKF loop
                eta_i = x_k1_k
                err = 2*self.EPSILON
                iter = 0

                while err > self.EPSILON:
                    if iter > self.MAX_ITER:
                        print("Terminating IEKF - exceeded maximum iterations")
                        break

                    iter += 1
                    eta1 = eta_i

                    # 5. recalculate Jacobian of measurement equation Hx
                    Hx = self.Hx(0, eta1, self.Uk[k])
                    z_k1_k = self.h(0, eta1, self.Uk[k])

                    # 6. Kalman gain (K_k1) recalculation
                    K_k1 = P_k1_k @ Hx.T @ np.linalg.inv(Hx @ P_k1_k @ Hx.T + self.R)

                    # 7. update measurement and state estimates x_k1_k1
                    eta_i = x_k1_k + K_k1 @ (self.Zk[k] - z_k1_k - Hx @ (x_k1_k - eta1))
                    eta_i = np.ravel(eta_i)
                    err = np.linalg.norm(eta_i - eta1) / np.linalg.norm(eta1)

                x_k1_k1 = eta_i

                # 8. update state estimation error covariance matrix P_k1_k1
                P_k1_k1 = (np.eye(self.states) - K_k1 @ Hx) @ P_k1_k @ (np.eye(self.states) - K_k1 @ Hx).T + K_k1 @ self.R @ K_k1.T

                # store results
                self.Xk1k1[:, k] = x_k1_k1
                self.Pk1k1[:, :, k] = P_k1_k1
                self.Z_pred[:, k] = z_k1_k
                pbar.update(1)

        # save Cm measurement and predicted outputs to file
        np.savetxt('data/output.csv', np.hstack([self.Cm.reshape(-1, 1), self.Z_pred.T]), delimiter=',')
        print("State estimation complete. The final value of C_alpha_up is: ", self.Xk1k1[3, -1])

    def alpha_reconstruction(self):
        """Reconstruction of alpha_true using the estimated C_alpha_up state : P1.5"""
        # α_true = α_meas / (1 + C_α_up)
        alpha_recon = self.Z_pred[0, :].copy()
        alpha_recon /= (1 + self.Xk1k1[3, :])
        return alpha_recon


    def plot(self, show=False):
        """Plot the results of the IEKF."""
        T = np.linspace(0, 100, 10001)

        plt.figure(figsize=(7, 3))
        plt.plot(self.Xk1k1[3, :], lw=1)
        plt.grid()
        plt.xlabel('Iteration [-]')
        plt.ylabel(r'$C_{\alpha_{up}}$ [-]')
        plt.title(r'Estimated $C_{\alpha_{up}}$ state')
        plt.tight_layout()
        plt.savefig('plots/Caup.png', dpi=300)
        if show:
            plt.show()

        labels = [r'$\alpha$ [rad]', r'$\beta$ [rad]', r'$V$ [m/s]']
        fig, ax = plt.subplots(3,1, figsize=(7, 8))
        ax[0].set_title('IEKF output estimation')
        for i in range(3):
            ax[i].plot(T, self.Zk[:, i], label='Measurement', lw=1.5)
            ax[i].plot(T, self.Z_pred[i, :], label='Predicted', lw=1.3, c='r')
            ax[i].legend()
            ax[i].grid()
            ax[i].set_xlabel('Time [s]')
            ax[i].set_ylabel(fr'{labels[i]}')
        ax[0].plot(T, self.alpha_reconstruction(), label=r'$\alpha_{true}$ Reconstruction', lw=1.3, c='g')
        ax[0].legend()
        plt.tight_layout()
        plt.savefig('plots/IEKF.png', dpi=300)
        if show:
            plt.show()

        plt.figure(figsize=(8, 6))
        plt.plot(self.Zk[:, 0], self.Zk[:, 1], label='Measured', lw=0.8)
        plt.plot(self.Z_pred[0, :], self.Z_pred[1, :], label='Predicted', lw=1, c='r')
        plt.plot(self.alpha_reconstruction(), self.Z_pred[1, :], label=r'$\alpha_{true}$ Reconstruction', lw=1, c='g')
        plt.legend()
        plt.grid()
        plt.xlabel(r'$\alpha$ [rad]')
        plt.ylabel(r'$\beta$ [rad]')
        plt.title(r'F16 $C_m(\alpha, \beta)$')
        plt.tight_layout()
        plt.savefig('plots/a_b.png', dpi=300)
        if show:
            plt.show()
