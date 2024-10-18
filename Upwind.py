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
    init = initial_condition2

    h = 0.01
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

    """ PLOTTING Solutions
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
        figure.canvas.flush_events()"""

    ## Calculating and Plotting Error
    hvals = (.1, .05, 0.025, 0.0125, 0.00625, 0.003125, 0.0015625, 0.00078125)
    hs = -1
    maxerror = np.zeros((len(hvals), 3))

    for h in hvals:
        hs = hs+1
        k = h / 2

        Nx = int((xbounds[1] - xbounds[0]) / h) + 1
        Nt = int((tbounds[1] - tbounds[0]) / k) + 1
        x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
        t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

        solA, solB = upwind(A1p, A1n, init, h, k, xbounds, tbounds)


        error_1 = np.zeros(len(t))
        error_2 = np.zeros(len(t))
        error_inf = np.zeros(len(t))

        for i in range(0, Nt):
            sol = np.array((solA[:,i], solB[:,i]))
            AnaA = analytical1(x, t[i], init)[0]
            AnaB = analytical1(x, t[i], init)[1]
            analytical = np.array((AnaA, AnaB))

            temperr = np.zeros(len(x))
            for j in range(0, Nx):
                temp = abs(analytical[:, j] - sol[:, j])
                error_1[i] = error_1[i] + (abs(temp[0]) + abs(temp[1]))
                error_2[i] = error_2[i] + pow((pow(temp[0], 2) + pow(temp[1], 2)), 1/2)
                temperr[j] = max(abs(temp))

            error_inf[i] = max(temperr)
        error_1 = error_1 * h
        error_2 = error_2 * h

        maxerror[hs, 0] = max(error_1)
        maxerror[hs, 1] = max(error_2)
        maxerror[hs, 2] = max(error_inf)

        print("\nMax Error when h = %f" % hvals[hs])
        print("Upwind Errors:")
        for i in (0, 1):
            print("\t e_%d = %f" % (i + 1, maxerror[hs, i]))
        print("\t e_inf = %f" % maxerror[hs, 2])

    logH = np.log(hvals)

    figure, axis = plt.subplots(2, 2)
    for i in (0, 1, 2):
        figplace1 = (0, 0, 1, 1)
        figplace2 = (0, 1, 0, 1)
        slope = (-np.log(maxerror[len(hvals)-1, i]) + np.log(maxerror[len(hvals)-2, i])) / (-logH[len(hvals)-1] + logH[len(hvals)-2])

        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror[:, i]),
                                              label="Error - Slope %f" % slope)

        axis[figplace1[i], figplace2[i]].legend()
        if i != 2:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
        else:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

    plt.show()



