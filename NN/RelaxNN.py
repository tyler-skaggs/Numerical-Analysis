import sys

sys.path.insert(0, '../Utilities/')

import torch
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import warnings
import time
from itertools import chain

warnings.filterwarnings('ignore')
np.random.seed(1324)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
else:
    device = torch.device('cpu')
    print('Using CPU')

# The Deep Neural Network

class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers)-1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy Layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

# The Relaxed physics-guided nerual network
class RelaxNN():
    def __init__(self, X_init, u_init, X, layers_u, layers_v, lb, ub, nu = 1):
        # Boundary Conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        self.t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)
        self.x_init = torch.tensor(X_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(X_init[:, 1:2], requires_grad=True).float().to(device)

        self.u_init = torch.tensor(u_init).float().to(device)

        self.layers_u = layers_u
        self.layers_v = layers_v
        self.nu = nu

        # Deep Neural Networks
        self.dnn_u = DNN(layers_u).to(device)
        self.dnn_v = DNN(layers_v).to(device)

        self.optimizer = torch.optim.Adam(chain(self.dnn_u.parameters(), self.dnn_v.parameters()))
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn_u(torch.cat([x, t], dim=1))
        return u

    def net_v(self, x, t):
        v = self.dnn_v(torch.cat([x, t], dim=1))
        return v

    def net_res(self, x, t):
        """ The pytorch autograd for calculating residual """
        u = self.net_u(x, t)
        v = self.net_v(x, t)

        u_t = torch.autograd.grad(
            u, t,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]

        v_x = torch.autograd.grad(
            v, x,
            grad_outputs=torch.ones_like(v),
            retain_graph=True,
            create_graph=True
        )[0]

        return u_t + v_x

    def loss_func(self):
        omega = np.array([0.1, 2, 10])
        #omega = np.array([0.5, 2, 5])

        u_IC_pred = self.net_u(self.x_init, self.t_init)
        u_pred = self.net_u(self.x, self.t)
        v_pred = self.net_v(self.x, self.t)
        res_pred = self.net_res(self.x, self.t)

        loss_IC = torch.mean((self.u_init - u_IC_pred) ** 2)
        loss_res = torch.mean(res_pred**2)
        loss_flux = torch.mean((v_pred - 1/2 * u_pred**2) ** 2)

        loss = omega[0] * loss_res + omega[1] * loss_flux + omega[2] * loss_IC

        return loss

    def train(self, nIter):
        for epoch in range(nIter):
            # Zero Gradients
            self.optimizer.zero_grad()

            # Compute Loss and Gradients
            loss = self.loss_func()
            loss.backward()

            #Adjust Learning Weights
            self.optimizer.step()

            if epoch % 100 == 0:
                print(
                    'Epoch %d, Loss: %.5e' % (
                        epoch,
                        loss.item()
                    )
                )

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn_u.eval()
        self.dnn_v.eval()
        u = self.net_u(x, t)
        res = self.net_res(x, t)
        u = u.detach().cpu().numpy()
        res = res.detach().cpu().numpy()

        return u, res

def analytical1(x, t):
    y = np.zeros(np.size(x))
    ul = 0
    ur = 1
    for i in range(0, np.size(x)):
        if x[i] <= ul * t:
            y[i] = ul
        elif x[i] > t * ur:
            y[i] = ur
        else:
            y[i] = x[i] / t
    return y

def analytical2(x, t):
    y = np.zeros(np.size(x))
    ul = 1
    ur = 0
    s = (ul + ur)/2

    for i in range(0, np.size(x)):
        if x[i] <= s * t:
            y[i] = ul
        else:
            y[i] = ur
    return y

def analytical3(x, t):
    y = np.zeros(np.size(x))

    for i in range(0, np.size(x)):
        if t < 2:
            if x[i] <= t * 3 / 2:
                y[i] = 2
            elif x[i] < 1/2 * t + 2:
                y[i] = 1
            else:
                y[i] = 0
        if t >= 2:
            if x[i] <= t + 1:
                y[i] = 2
            else:
                y[i] = 0
    return y

#------------------------------------------------------------

def init(x):
    #return -np.sin(np.pi * x)
    return analytical2(x,0)

analytic_OffOn = 1
def analytic_Sol(x,t):
    return analytical2(x, t)


nu = 0.01/np.pi
Nx = 100
Nt = 50
N_f = 10000
epochs = 20000

layers_u = [2, 128, 128, 128, 128, 1]
layers_v = [2, 64, 64, 64, 64, 1]

lb = -1
ub = 1
lb_uval = 1
ub_uval = 0

plot_low = -0.2
plot_high = 1.2

# (x, t = 0)
x = 2 * np.random.random_sample(Nx) - 1
init_x = np.vstack((x,np.zeros(len(x)))).T
u_init = init(x)


# Lower x Bound
t = np.random.random_sample(Nt)
init_lb = np.vstack((lb * np.ones(Nt), t)).T
u_lb = lb_uval * np.ones(Nt)

# Upper x Bound
t = np.random.random_sample(Nt)
init_ub = np.vstack((ub * np.ones(Nt), t)).T
u_ub = ub_uval * np.ones(Nt)

X_initial = np.vstack((init_x, init_lb, init_ub))
u_initial = np.array([np.hstack((u_init, u_lb, u_ub))]).T

# Random 10000 data to train on
X_training = np.random.random_sample((N_f, 2))
X_training[:, 0] = 2 * X_training[:, 0] - 1
x = X_training[:, 0]
t = X_training[:, 1]

X_training = np.vstack((X_training, X_initial))

# training
start_time = time.time()

model = RelaxNN(X_initial, u_initial, X_training, layers_u, layers_v, lb, ub)
model.train(epochs)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Executation Time: {elapsed_time:.6f} seconds")


#---------------------------------- PLOTTING -------------------------------------

x_pred = np.array([np.linspace(-1,1, 100)]).T
t_pred = np.array([np.linspace(0,1, 101)]).T
X, T = np.meshgrid(x_pred,t_pred)

X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

u_pred, f_pred = model.predict(X_pred)
U_pred = griddata(X_pred, u_pred.flatten(), (X, T), method = 'cubic')


####### HEAT MAP AND SLICES ##################
""" The aesthetic setting has changed. """
dx = x[1]-x[0]

fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111)
gs1 = gridspec.GridSpec(2, 4)
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

ax.plot(
    X_initial[:,1],
    X_initial[:,0],
    'kx', label = 'Data (%d points)' % (u_initial.shape[0]),
    markersize = 4,  # marker size doubled
    clip_on = False,
    alpha=.5
)

line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(0.25*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.5*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(0.75*np.ones((2,1)), line, 'w-', linewidth = 1)

ax.set_xlabel('$t$', size=20)
ax.set_ylabel('$x$', size=20)
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.9, -0.05),
    ncol=5,
    frameon=False,
    prop={'size': 15}
)
ax.set_title('$u(t,x)$', fontsize = 20) # font size doubled
ax.tick_params(labelsize=15)

# Slices
ax = plt.subplot(gs1[1, 0])
if analytic_OffOn:
    ax.plot(x_pred, analytic_Sol(x_pred, t = 0.25), 'b-', linewidth = 2, label = 'Exact')
    error = pow(dx * sum(pow((analytic_Sol(x_pred, t=0.25) - U_pred[25, :]), 2)), 1 / 2)
    text = plt.text(-1, 0, f"Error = %.5f" % error)
ax.plot(x_pred, U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize = 15)
ax.axis('square')
ax.set_xlim([lb-0.1, ub+0.1])
ax.set_ylim([plot_low, plot_high])



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[1, 1])
if analytic_OffOn:
    ax.plot(x_pred, analytic_Sol(x_pred, t = 0.5), 'b-', linewidth = 2, label = 'Exact')
    error = pow(dx * sum(pow((analytic_Sol(x_pred, t=0.5) - U_pred[50, :]), 2)), 1 / 2)
    text = plt.text(-1, 0, f"Error = %.5f" % error)
ax.plot(x_pred, U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([lb-0.1, ub+0.1])
ax.set_ylim([plot_low, plot_high])
ax.set_title('$t = 0.50$', fontsize = 15)



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

ax = plt.subplot(gs1[1, 2])
if analytic_OffOn:
    ax.plot(x_pred, analytic_Sol(x_pred, t = 0.75), 'b-', linewidth = 2, label = 'Exact')
    error = pow(dx * sum(pow((analytic_Sol(x_pred, t=0.75) - U_pred[75, :]), 2)), 1 / 2)
    text = plt.text(-1, 0, f"Error = %.5f" % error)
ax.plot(x_pred,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([lb-0.1, ub+0.1])
ax.set_ylim([plot_low, plot_high])
ax.set_title('$t = 0.75$', fontsize = 15)



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[1, 3])
if analytic_OffOn:
    ax.plot(x_pred, analytic_Sol(x_pred, t = 1), 'b-', linewidth = 2, label = 'Exact')
    error = pow(dx * sum(pow((analytic_Sol(x_pred, t=1) - U_pred[100, :]), 2)), 1 / 2)
    text = plt.text(-1, 0, f"Error = %.5f" % error)
ax.plot(x_pred,U_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([lb-0.1, ub+0.1])
ax.set_ylim([plot_low, plot_high])
ax.set_title('$t = 1.0$', fontsize = 15)



for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()


""" The aesthetic setting has changed. """
####### Row 1: u(t,x) Animated ##################
plt.ion()
figure = plt.figure()
axis = figure.add_subplot(111)

if analytic_OffOn:
    line0, = axis.plot(x_pred, init(x_pred), 'red', label='Analytical Solution')
linePINN, = axis.plot(x_pred, init(x_pred), color='purple', label='PINN Solution')

plt.ylim(plot_low, plot_high)
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")

i = 0
time = 1
tt = 0
dt = t_pred[1]-t_pred[0]
text = plt.text(0, 0, "t = 0")

while tt < time - dt/2:
    if analytic_OffOn:
        line0.set_ydata(analytic_Sol(x_pred, tt))
    linePINN.set_ydata(U_pred[i, :])

    figure.canvas.draw()
    figure.canvas.flush_events()
    tt += dt
    text.set_text("t = %f" % tt)
    i += 1

plt.ioff()
plt.show()

#print('Error u: %e' % (error_u))


