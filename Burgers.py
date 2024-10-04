import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import length


def f(x):
    return pow(x, 2) / 2


def Richtmyer_Two_Step(k, h, init, xbound, tbound):
    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)
    temp = init(x)
    sol = np.zeros((Nx, Nt))
    for t in range(0, Nt):
        sol[:, t] = y
        temp[1:-1] = 1 / 2 * ( y[1:-1] + y[2:] ) - k / (2 * h) * (f(y[2:]) - f(y[1:-1]))
        y[1:-1] = y[1:-1] - k/h * ( f(temp[1:-1]) - f(temp[:-2]) )
        #temp1 = 1 / 2 * ( y[1:-1] + y[2:] ) - k / (2 * h) * (f(y[2:]) - f(y[1:-1]))
        #temp2 = 1 / 2 * ( y[:-2] + y[1:-1] ) - k / (2 * h) * (f(y[1:-1]) - f(y[:-2]))
        #for i in range(1, Nx-1):
            #y[i] = y[i] - k / h * (f(temp1[i-1]) - f(temp2[i-1]))

    return sol


def MacCormack(k, h, init, xbound, tbound):
    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)
    temp = init(x)

    sol = np.zeros((Nx, Nt))
    for t in range(0, Nt):
        sol[:, t] = y
        temp[1:-1] = y[1:-1] - k / h * (f(y[1:-1]) - f(y[:-2]))

        y[1:-1] = 1/2 * (y[1:-1] + temp[1:-1]) - k/(2*h) * (f(temp[1:-1]) - f(temp[:-2]))

    return sol


def LaxFriedrichs(k, h, init, xbound, tbound):
    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)

    sol = np.zeros((Nx, Nt))

    for t in range(0, Nt):
        sol[:, t] = y
        y[1:-1] = 1/2 * (y[:-2] + y[2:]) - k / (2 * h) * (f(y[2:]) - f(y[:-2]))

    return sol


def initial_condition1(x):
    y = np.zeros(np.size(x))
    ul = 0
    ur = 1
    for i in range(0, np.size(x)):
        if(x[i] >= 0.5):
            y[i] = ur
        elif(x[i] < 0.5):
            y[i] = ul
    return(y)


def analytical1(x, t):
    y = np.zeros(np.size(x))
    x = x - 0.5
    ul = 0
    ur = 1
    for i in range(0, np.size(x)):
        if x[i] < ul * t:
            y[i] = ul
        elif x[i] <= t*ur:
            y[i] = x[i] / t
        else:
            y[i] = ur
    return y


def analytical2(x, t):
    y = np.zeros(np.size(x))
    x = x - 0.5
    ul = 1
    ur = 0
    s = (ul + ur)/2

    for i in range(0, np.size(x)):
        if x[i] < s * t:
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


def initial_condition3(x):
    return analytical3(x, 0)


if __name__ == '__main__':
    h = 0.01
    k = 0.005
    xbounds = (0, 6)
    tbounds = (0, 4)
    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1
    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

    sol1 = LaxFriedrichs(k, h, initial_condition3, xbounds, tbounds)
    sol2 = Richtmyer_Two_Step(k, h, initial_condition3, xbounds, tbounds)
    sol3 = MacCormack(k, h, initial_condition3, xbounds, tbounds)

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    line0, = axis.plot(x, initial_condition3(x), 'red', label='Analytical Solution')
    line1, = axis.plot(x, sol1[:, 0], 'blue', label='Lax-Wendroff')  # Returns a tuple of line objects, thus the comma
    line2, = axis.plot(x, sol2[:, 0], 'green', label='Richtmyer')  # Returns a tuple of line objects, thus the comma
    line3, = axis.plot(x, sol3[:, 0], 'orange', label='MacCormack')  # Returns a tuple of line objects, thus the comma

    plt.legend()
    plt.ylabel("u(x)")
    plt.xlabel("x")
    plt.title(r"Burgers' Equation")
    plt.ylim(min(sol1[:, 0]) - 0.2, max(sol1[:,0]) + 0.2)

    for i in range(1, Nt):
        line0.set_ydata(analytical3(x, t[i]))
        line1.set_ydata(sol1[:, i])
        line2.set_ydata(sol2[:, i])
        line3.set_ydata(sol3[:, i])
        figure.canvas.draw()
        figure.canvas.flush_events()

