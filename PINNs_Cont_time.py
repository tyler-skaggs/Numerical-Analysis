import sys

from sqlalchemy.testing.exclusions import requires_tag

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

# The physics-guided nerual network
class PhysicsInformedNN():
    def __init__(self, X_init, u_init, X_f, layers, lb, ub, nu):
        # Boundary Conditions
        self.lb = torch.tensor(lb).float().to(device)
        self.ub = torch.tensor(ub).float().to(device)

        # data
        self.x_f = torch.tensor(X_f[:, 0:1], requires_grad=True).float().to(device)
        self.t_f = torch.tensor(X_f[:, 1:2], requires_grad=True).float().to(device)
        self.x_init = torch.tensor(X_init[:, 0:1], requires_grad=True).float().to(device)
        self.t_init = torch.tensor(X_init[:, 1:2], requires_grad=True).float().to(device)

        self.u_init = torch.tensor(u_init).float().to(device)

        self.layers = layers
        self.nu = nu

        # settings
        #self.lambda_1 = torch.tensor([0.0], requires_grad=True).to(device)
        #self.lambda_2 = torch.tensor([-6.0], requires_grad=True).to(device)

        #self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        #self.lambda_2 = torch.nn.Parameter(self.lambda_2)

        # Deep Neural Networks
        self.dnn = DNN(layers).to(device)
        #self.dnn.register_parameter('lambda_1', self.lambda_1)
        #self.dnn.register_parameter('lambda_2', self.lambda_2)

        # optimizer: using the same settings
        self.optimizer = torch.optim.LBFGS(
            self.dnn.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=0.00001,
            tolerance_change=1.0*np.finfo(float).eps, #machine epsilon
            line_search_fn="strong_wolfe"
        )

        #self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())
        self.iter = 0

    def net_u(self, x, t):
        u = self.dnn(torch.cat([x, t], dim=1))
        return u

    def net_f(self, x, t):
        """ The pytorch autograd for calculating residual """
        u = self.net_u(x, t)

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

        u_xx = torch.autograd.grad(
            u_x, x,
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0]

        f = u_t + u * u_x - self.nu * u_xx
        return f

    def loss_func(self):
        self.optimizer.zero_grad()

        u_pred = self.net_u(self.x_init, self.t_init)
        f_pred = self.net_f(self.x_f, self.t_f)

        loss_u = torch.mean((self.u_init - u_pred) ** 2)
        loss_f = torch.mean(f_pred ** 2)

        loss = loss_u + loss_f

        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            print(
                'Iter %d, Loss: %.5e, Loss_u: %.5e, Loss_f: %.5e' % (
                    self.iter, loss.item(), loss_u.item(), loss_f.item()
                )
            )

        return loss

    def train(self):
        self.dnn.train()
        print("Iter %d" % self.iter)

        # Backward and optimize
        self.optimizer.step(self.loss_func)

    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(device)

        self.dnn.eval()
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        return u, f


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

def init(x):
    return -np.sin(np.pi * x)



nu = 0.01/np.pi
Nx = 50
Nt = 25
N_f = 10000
layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

lb = -1
ub = 1
lb_uval = 0
ub_uval = 0

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
model = PhysicsInformedNN(X_initial, u_initial, X_training, layers, lb, ub, nu=0.01/np.pi)
model.train()

x_pred = np.array([np.linspace(-1,1, 256)]).T
t_pred = np.array([np.linspace(0,1, 101)]).T
X, T = np.meshgrid(x_pred,t_pred)

X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

u_pred, f_pred = model.predict(X_pred)
U_pred = griddata(X_pred, u_pred.flatten(), (X, T), method = 'cubic')


""" The aesthetic setting has changed. """
####### Row 0: u(t,x) ##################

fig = plt.figure(figsize=(9, 5))
ax = fig.add_subplot(111)

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

plt.show()

####### Row 1: u(t,x) slices ##################
plt.ion()
figure = plt.figure()
axis = figure.add_subplot(111)

#line0, = axis.plot(x_pred, init(x_pred), 'red', label='Analytical Solution')
linePINN, = axis.plot(x_pred, init(x_pred), color='purple', label='PINN Solution')

plt.ylim(min(init(x_pred))[0]-0.2, max(init(x_pred))[0]+0.2)
plt.legend()
plt.xlabel("x")
plt.ylabel("u(x,t)")

i = 0
time = 1
t = 0
dt = t_pred[1]-t_pred[0]
text = plt.text(0, 0, "t = 0")

while t < time - dt/2:
    #line0.set_ydata(analytical2(x_pred, t))
    linePINN.set_ydata(U_pred[i, :])

    figure.canvas.draw()
    figure.canvas.flush_events()
    t += dt
    text.set_text("t = %f" % t)
    i += 1

plt.ioff()
plt.show()


""" The aesthetic setting has changed. """

"""fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111)

gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)

ax = plt.subplot(gs1[0, 0])
#ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x_pred,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.set_title('$t = 0.25$', fontsize = 15)
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

ax = plt.subplot(gs1[0, 1])
#ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x_pred,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
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

ax = plt.subplot(gs1[0, 2])
#ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')
ax.plot(x_pred,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$u(t,x)$')
ax.axis('square')
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 0.75$', fontsize = 15)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
    item.set_fontsize(15)

plt.show()"""

# evaluations
#u_pred, f_pred = model.predict(X_star)

#lambda_1_value_noisy = model.lambda_1.detach().cpu().numpy()
#lambda_2_value_noisy = model.lambda_2.detach().cpu().numpy()
#lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

#error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
#error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

#print('Error u: %e' % (error_u))
#print('Error l1: %.5f%%' % (error_lambda_1_noisy))
#print('Error l2: %.5f%%' % (error_lambda_2_noisy))
