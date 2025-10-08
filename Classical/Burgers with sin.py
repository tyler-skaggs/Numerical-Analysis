import numpy as np
import matplotlib.pyplot as plt
from numpy import where
import scipy.optimize as opt
from Solvers import solver

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def initial_sin(x):
    y = np.zeros(np.size(x))
    for i in range(0,len(x)):
        if x[i] == 0 or x[i] ==1:
            y[i] = 0
        else:
            y[i] = np.sin(2 * np.pi * x[i])

    return y

if __name__ == '__main__':
    init = initial_sin
    problem = burgers
    deriv = burgers_prime

    bound = 1

    h = 0.005
    k = h / 3
    xbounds = (0, 1)
    tbounds = (0, 1)
    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1
    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

    sol = solver(k, h, init, xbounds, tbounds, problem, deriv, "G")

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    line1, = axis.plot(x, sol[:, 0], color='green', label='Numerical Solutions')

    plt.legend()
    plt.xlabel("x")
    if problem == burgers:
        plt.title(r"Burger's Equations")
        plt.ylabel(r"$u(x,t)$")

    plt.ylim(min(sol[:, 0]) - 0.2, max(sol[:, 0]) + 0.2)

    text = plt.text(0, 0, "t = 0")

    plt.axvline(x=1/2, color='red', label='Analytic Location')
    found = 0
    for i in range(1, Nt):
        text.set_text("t = %f" % t[i])
        line1.set_ydata(sol[:, i])
        figure.canvas.draw()
        figure.canvas.flush_events()
        if not found:
            for j in range(0, Nx-2):
                if abs(sol[j, i] - sol[j+2,i]) >= bound:
                    print("Discontinuity: \n\t time t = %f \n\t Location x = %f" % (t[i], x[j+1]))
                    plt.axvline(x=x[j+1], color='blue', label='Numerical Location')
                    found = 1

