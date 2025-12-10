import sys
from itertools import chain

sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy import where
import scipy.io
from scipy.stats import qmc
import scipy.integrate as integrate
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import time

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

warnings.filterwarnings('ignore')

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')

class DNN(torch.nn.Module):
    def __init__(self, layers, custom_activ):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers)-1

        # set up layer order dict
        self.activation = custom_activ()

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy Layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

# The physics-guided nerual network
class PhysicsInformedNN():
    def __init__(self, X_IC_init, X_BC_init,  X_f, layers, lb, ub, A, custom_activ = torch.nn.Tanh):
        # Boundary Conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        self.A = A

        # data
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)

        self.x_IC = torch.tensor(X_IC_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_IC = torch.tensor(X_IC_init[:, 1:2], requires_grad=True).float().to(device)

        self.x_BC = torch.tensor(X_BC_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_BC = torch.tensor(X_BC_init[:, 1:2], requires_grad=True).float().to(device)

        self.u_IC = torch.tensor(X_IC_init[:, 2:3]).float().to(device)
        self.u_BC = torch.tensor(X_BC_init[:, 2:3]).float().to(device)

        self.v_IC = torch.tensor(X_IC_init[:, 3:4]).float().to(device)
        self.v_BC = torch.tensor(X_BC_init[:, 3:4]).float().to(device)

        self.layers = layers

        # Deep Neural Networks
        self.dnn = DNN(layers, custom_activ).to(device)

        # optimizer: using the same settings
        self.optimizer = torch.optim.Adam( self.dnn.parameters() )

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))[:, 0:1]
        return u

    def net_v(self, x, t):
        v = self.dnn(torch.cat([x, t], dim=1))[:, 1:2]
        return v


    def net_f(self, x, t):
        """ The pytorch autograd for calculating residual """
        u = self.net_u(x, t)
        v = self.net_v(x,t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        v_t = torch.autograd.grad(
            v, t,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f1 = u_t - 2 * u_x + v_x
        f2 = v_t + 2*v_x
        return f1, f2

    def loss_func(self):
        omega = np.array([1, 5, 5])
        omega = omega/sum(omega)

        u_BC_pred = self.net_u(self.x_BC, self.t_BC)
        u_IC_pred = self.net_u(self.x_IC, self.t_IC)

        v_BC_pred = self.net_v(self.x_BC, self.t_BC)
        v_IC_pred = self.net_v(self.x_IC, self.t_IC)

        f1_pred, f2_pred = self.net_f(self.x_f, self.t_f)

        loss_BC = torch.nn.MSELoss()(u_BC_pred, self.u_BC) + torch.nn.MSELoss()(v_BC_pred, self.v_BC)
        loss_IC = torch.nn.MSELoss()(u_IC_pred, self.u_IC) + torch.nn.MSELoss()(v_IC_pred, self.v_IC)

        loss_PDE = torch.mean(f1_pred ** 2) + torch.mean(f2_pred ** 2)

        loss = omega[0]*loss_PDE + omega[1] * loss_IC + omega[2] * loss_BC

        return loss, loss_PDE, loss_IC, loss_BC

    def train(self, epochs):
        loss_hist = np.array([0,0,0,0])
        for epoch in range(epochs):
            # Zero Gradients
            self.optimizer.zero_grad()

            # Compute Loss and Gradients
            loss, loss_PDE, loss_IC, loss_BC = self.loss_func()
            loss.backward()

            # Adjust Learning Weights
            self.optimizer.step()

            if epoch % 100 == 0:
                print(
                    'Epoch %d | Loss: %.5e, L_PDE: %.5e, L_IC: %.5e, L_BC: %.5e' % (
                        epoch,
                        loss.item(),
                        loss_PDE.item(),
                        loss_IC.item(),
                        loss_BC.item()
                    )
                )
            loss_hist = np.vstack((loss_hist, np.array(([
                loss.detach().cpu().numpy(),
                loss_PDE.detach().cpu().numpy(),
                loss_IC.detach().cpu().numpy(),
                loss_BC.detach().cpu().numpy()
            ]))))

        return np.delete(loss_hist, (0), axis=0)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        v = self.net_v(x, t)

        u = u.detach().cpu().numpy()
        v = v.detach().cpu().numpy()

        return u, v



def initial_condition1(x):
    y = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        if(x[i] >= 1.5):
            y[i] = 1
        elif(x[i] < 1.5):
            y[i] = 0
    return(y)


def initial_condition2(x):
    f = np.zeros_like(x)
    x_left = 1
    x_right = 3
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f


def analytical2(x, t, init):
    t1 = init(x + 2 * t) - 1/4 * init(x + 2 * t) + 1/4 * init(x - 2 * t)
    t2 = init(x - 2 * t)
    return t1, t2


class Cauchy_Activation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda1 = torch.nn.Parameter(torch.tensor(0.01))
        self.lambda2 = torch.nn.Parameter(torch.tensor(0.01))
        self.d = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return (self.lambda1 * x) / (x**2 + self.d**2) + self.lambda2 / (x**2 + self.d**2)


if __name__ == "__main__":

    init = initial_condition1
    analytic_Sol = lambda x, t: analytical2(x, t, init)

    A = np.matrix([[-2, 1], [0, 2]])

    lb = 0 #x bounds
    ub = 4

    u_lb_val = 0
    u_ub_val = 1
    v_lb_val = 0
    v_ub_val = 1

    plot_high = 1.5
    plot_low = -0.5

    TIME = 0.5

    epochs = 10000

    Nx = 100
    Nt = 100
    N_f = 250

    analytic_OffOn = 1

    layers = [2, 100, 100, 100, 100, 2]

    # (x, t = 0)
    x = (ub - lb) * np.random.random_sample(Nx) + lb

    X_IC = np.vstack((x, np.zeros(len(x)), init(x), init(x))).T

    # Lower x Bound
    t = np.random.random_sample(Nt) * TIME
    u_lb = u_lb_val * np.ones(len(t))
    v_lb = v_lb_val * np.ones(len(t))
    init_lb = np.vstack((lb * np.ones(Nt), t, u_lb, v_lb)).T

    # Upper x Bound
    t = np.random.random_sample(Nt) * TIME
    u_ub = u_ub_val * np.ones(len(t))
    v_ub = v_ub_val * np.ones(len(t))
    init_ub = np.vstack((ub * np.ones(Nt), t, u_ub, v_ub)).T

    X_BC = np.vstack((init_lb, init_ub))

    # Random N_f data to train on
    X_training = np.random.random_sample((N_f, 2))
    X_training[:, 0] = (ub - lb) * X_training[:, 0] + lb
    X_training[:, 1] = X_training[:, 1] * TIME
    x = X_training[:, 0]
    t = X_training[:, 1]
    X_training = np.vstack((X_training, X_IC[:, 0:2], X_BC[:, 0:2]))

    start_time = time.time()

    model = PhysicsInformedNN(X_IC, X_BC, X_training, layers, lb, ub, A, Cauchy_Activation)#torch.nn.Tanh)

    loss_history = model.train(epochs)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Executation Time: {elapsed_time:.6f} seconds")

    # Predictions
    x_pred = np.array([np.linspace(lb, ub, 101)]).T
    t_pred = np.array([np.linspace(0, TIME, 101)]).T
    X, T = np.meshgrid(x_pred, t_pred)

    X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    u_pred, v_pred = model.predict(X_pred)
    U_pred = griddata(X_pred, u_pred.flatten(), (X, T), method='cubic')
    V_pred = griddata(X_pred, v_pred.flatten(), (X, T), method='cubic')

    # -------------------------------------------------
    # ------------------ PLOTTING ---------------------
    # -------------------------------------------------

    # Loss Graph
    plt.plot(range(1001, epochs + 1), loss_history[1000:, 0], color='red', label=r"$L(\theta)$")
    plt.plot(range(1001, epochs + 1), loss_history[1000:, 1], color='blue', label=r"$\mathcal{L}_{PDE}$")
    plt.plot(range(1001, epochs + 1), loss_history[1000:, 2], color='orange', label=r"$\mathcal{L}_{IC}$")
    plt.plot(range(1001, epochs + 1), loss_history[1000:, 3], color='green', label=r"$\mathcal{L}_{BC}$")
    plt.ylabel(r"$\mathcal{L}$")
    plt.xlabel("Epoch")
    plt.legend()
    plt.yscale('log')
    plt.title("Loss History")
    plt.show()

    # Heat Map
    ####### HEAT MAP AND SLICES ##################
    """ The aesthetic setting has changed. """
    dx = x_pred[1] - x_pred[0]

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    gs1 = gridspec.GridSpec(4, 4)
    gs1.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.5)

    # HEAT MAP
    ax = plt.subplot(gs1[0, :])
    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    X_initial = np.vstack((X_IC, X_BC))
    ax.plot(
        X_initial[:, 1],
        X_initial[:, 0],
        'kx', label='Data (%d points)' % (X_initial[:, 2].shape[0]),
        markersize=4,  # marker size doubled
        clip_on=False,
        alpha=.5
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(.25 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.5 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.75 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$u(t,x)$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)

    # HEAT MAP 2
    ax = plt.subplot(gs1[1, :])
    h = ax.imshow(V_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    X_initial = np.vstack((X_IC, X_BC))
    ax.plot(
        X_initial[:, 1],
        X_initial[:, 0],
        'kx', label='Data (%d points)' % (X_initial[:, 2].shape[0]),
        markersize=4,  # marker size doubled
        clip_on=False,
        alpha=.5
    )

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(.25 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.5 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(.75 * TIME * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$', size=20)
    ax.set_ylabel('$x$', size=20)
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.9, -0.05),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )
    ax.set_title('$v(t,x)$', fontsize=20)  # font size doubled
    ax.tick_params(labelsize=15)

    # Slices u
    ax = plt.subplot(gs1[2, 0])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.25 * TIME)[0], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.25 * TIME)[0] - np.array([U_pred[25]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title(f"$t = %.2f$" % (0.25*TIME), fontsize=15)
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[2, 1])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.5 *TIME )[0], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.5 * TIME)[0] - np.array([U_pred[50]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % (0.5*TIME), fontsize=15)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[2, 2])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.75*TIME)[0], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.75*TIME)[0] - np.array([U_pred[75]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % (0.75*TIME), fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[2, 3])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=TIME)[0], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=TIME)[0] - np.array([U_pred[100]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, U_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % TIME, fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)


    # Slices v
    ax = plt.subplot(gs1[3, 0])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.25*TIME)[1], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.25*TIME)[1] - np.array([V_pred[25]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, V_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title(f"$t = %.2f$" % (0.25*TIME), fontsize=15)
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[3, 1])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.5*TIME)[1], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.5*TIME)[1] - np.array([V_pred[50]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, V_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % (0.5*TIME), fontsize=15)

    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        frameon=False,
        prop={'size': 15}
    )

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[3, 2])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=.75*TIME)[1], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=.75*TIME)[1] - np.array([V_pred[75]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, V_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % (0.75*TIME), fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    ax = plt.subplot(gs1[3, 3])
    if analytic_OffOn:
        ax.plot(x_pred, analytic_Sol(x_pred, t=TIME)[1], 'b-', linewidth=2, label='Exact')
        error = np.linalg.norm(analytic_Sol(x_pred, t=TIME)[1] - np.array([V_pred[100]]).T)
        text = plt.text(-1, -0.9, f"Error = %.5f" % error)
    ax.plot(x_pred, V_pred[100, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([lb - 0.1, ub + 0.1])
    ax.set_ylim([plot_low, plot_high])
    ax.set_title(f"$t = %.2f$" % TIME, fontsize=15)

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)

    plt.show()