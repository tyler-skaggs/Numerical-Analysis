import numpy as np
import scipy.io

def init(x):
    return -np.sin(np.pi * x)

data = scipy.io.loadmat('./burgers_shock.mat')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]

x_pred = np.array([np.linspace(-1,1, 256)]).T
t_pred = np.array([np.linspace(0,0.99, 100)]).T
X, T = np.meshgrid(x_pred,t_pred)


X_pred = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

print(min(init(x_pred))[0])