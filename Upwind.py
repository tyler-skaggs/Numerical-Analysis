import numpy as np
import matplotlib.pyplot as plt
from numpy import where


def upwind(Ap, An, init, h, k, xbound, tbound):

    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = np.matrix([init(x), init(x)])

    sol1 = np.zeros((Nx, Nt))
    sol2 = np.zeros((Nx, Nt))

    for t in range(0, Nt):
        sol1[:, t] = y[0]
        sol2[:, t] = y[1]

        y[:, 1:-1] = y[:, 1:-1] - (k / h) * Ap @ (y[:, 1:-1] - y[:, :-2]) - (k / h) * An @ (y[:, 2:] - y[:, 1:-1])

    return sol1, sol2


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
    x_left = 1.75
    x_right = 2.25
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f


def analytical1(x, t, init):
    t1 = init(x - t)
    t2 = init(x + t)
    return t1, t2


def analytical2(x, t, init):
    t1 = init(x + 2 * t) - 1/4 * init(x + 2 * t) + 1/4 * init(x - 2 * t)
    t2 = init(x - 2 * t)
    return t1, t2


if __name__ == '__main__':
    h = 0.001
    k = h/3
    xbounds = (0, 4)
    tbounds = (0, 1)
    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1
    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

    A1 = np.matrix([[1, 0], [0, -1]])
    A1p = np.matrix([[1, 0], [0, 0]])
    A1n = np.matrix([[0, 0], [0, -1]])

    A2 = np.matrix([[-2, 1], [0, 2]])
    Q = np.matrix([[0.25, 1], [1, 0]])
    Qinv = np.linalg.inv(Q)
    A2p = Q @ np.matrix([[2, 0], [0, 0]]) @ Qinv
    A2n = Q @ np.matrix([[0, 0], [0, -2]]) @ Qinv

    solA, solB = upwind(A1p, A1n, initial_condition2, h, k, xbounds, tbounds)

    AnaA, AnaB = analytical2(x, 0, initial_condition2)

    plt.ion()
    figure, axis = plt.subplots(2)

    line0, = axis[0].plot(x, AnaA, color='red', label='Analytical Solution')
    line1, = axis[1].plot(x, AnaB, color='red', label='Analytical Solution')

    line2, = axis[0].plot(x, solA[:, 0], color='blue', label="Numerical Solution")  # Returns a tuple of line objects, thus the comma
    axis[0].title.set_text('Equation A')

    line3, = axis[1].plot(x, solB[:, 0], color='blue', label="Numerical Solution")  # Returns a tuple of line objects, thus the comma
    axis[1].title.set_text('Equation B')

    plt.legend()

    for i in range(0, Nt):
        line0.set_ydata(analytical1(x, t[i], initial_condition2)[0])
        line1.set_ydata(analytical1(x, t[i], initial_condition2)[1])
        line2.set_ydata(solA[:, i])
        line3.set_ydata(solB[:, i])

        figure.canvas.draw()
        figure.canvas.flush_events()
