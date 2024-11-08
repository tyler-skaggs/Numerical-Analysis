import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.units import length


def f(x):
    umax = 1
    pmax = 2
    return x * umax * (1 - x / pmax)
    #return pow(x, 2) / 2

def Upwind(k, h, init, xbound, tbound):
    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)
    sol = np.zeros((Nx, Nt))

    def Flux(v, w):
        F = np.zeros(len(v))

        for i in range(0, len(v)):
            if v[i] == w[i]:
                F[i] = f(v[i])
            else:
                temp = (f(v[i]) - f(w[i])) / (v[i] - w[i])
                if temp >= 0:
                    F[i] = f(v[i])
                else:
                    F[i] = f(w[i])
        return F

    for t in range(0, Nt):
        sol[:, t] = y
        y[1:-1] = y[1:-1] - k / h * (Flux(y[1:-1], y[2:]) - Flux(y[:-2], y[1:-1]))

    return sol


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
    ul = 2
    ur = 0
    for i in range(0, np.size(x)):
        if(x[i] > 0):
            y[i] = ur
        elif(x[i] < 0):
            y[i] = ul
        else:
            y[i] = 1
    return(y)


def analytical1(x, t):
    y = np.zeros(np.size(x))
    x = x - 2
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

def analytical_Traffic(x, t):
    y = np.zeros(np.size(x))
    pmax = 2
    umax = 1
    for i in range(0, np.size(x)):
        if x[i] < -umax * t:
            y[i] = pmax
        elif x[i] >= umax * t:
            y[i] = 0
        else:
            y[i] = pmax / 2 * (1 - x[i] / (umax * t))

    return y

def initial_condition3(x):
    return analytical3(x, 0)


if __name__ == '__main__':
    init = initial_condition1

    h = 0.01
    k = 0.005
    xbounds = (-3, 3)
    tbounds = (0, 1)
    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1
    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time


    """sol1 = LaxFriedrichs(k, h, init, xbounds, tbounds)
    sol2 = Richtmyer_Two_Step(k, h, init, xbounds, tbounds)
    sol3 = MacCormack(k, h, init, xbounds, tbounds)
    sol4 = Upwind(k, h, init, xbounds, tbounds)

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')
    line1, = axis.plot(x, sol1[:, 0], 'blue', label='Lax-Friedrichs')  # Returns a tuple of line objects, thus the comma
    line2, = axis.plot(x, sol2[:, 0], 'green', label='Richtmyer')  # Returns a tuple of line objects, thus the comma
    #line3, = axis.plot(x, sol3[:, 0], 'orange', label='MacCormack')  # Returns a tuple of line objects, thus the comma
    line4, = axis.plot(x, sol4[:, 0], 'yellow', label='Upwind')  # Returns a tuple of line objects, thus the comma

    plt.legend()
    plt.ylabel(r"$\rho(x,t)$")
    plt.xlabel("x")
    plt.title(r"Green Light Problem $\rho(x,t)$")
    plt.ylim(min(sol1[:, 0]) - 0.2, max(sol1[:,0]) + 0.2)

    text = plt.text(-5, 0, "t = 0")

    for i in range(1, Nt):
        text.set_text("t = %f" % t[i])
        line0.set_ydata(analytical_Traffic(x, t[i]))
        line1.set_ydata(sol1[:, i])
        line2.set_ydata(sol2[:, i])
        #line3.set_ydata(sol3[:, i])
        line4.set_ydata(sol4[:, i])
        figure.canvas.draw()
        figure.canvas.flush_events()
"""
    ## Calculating and Plotting Error
    hvals = (.1, .05, 0.025, 0.0125, 0.00625, 0.003125)  # , 0.00156253, 0.00078125)
    hs = -1

    maxerror_u = np.zeros((len(hvals), 3))
    maxerror_lf = np.zeros((len(hvals), 3))
    maxerror_R = np.zeros((len(hvals), 3))

    for h in hvals:
        hs = hs + 1
        k = h / 3

        Nx = int((xbounds[1] - xbounds[0]) / h) + 1
        Nt = int((tbounds[1] - tbounds[0]) / k) + 1
        x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
        t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time


        error_1_u = np.zeros(len(t))
        error_2_u = np.zeros(len(t))
        error_inf_u = np.zeros(len(t))

        error_1_R = np.zeros(len(t))
        error_2_R = np.zeros(len(t))
        error_inf_R = np.zeros(len(t))

        error_1_lf = np.zeros(len(t))
        error_2_lf = np.zeros(len(t))
        error_inf_lf = np.zeros(len(t))


        sol_lf = LaxFriedrichs(k, h, init, xbounds, tbounds)
        sol_R = Richtmyer_Two_Step(k, h, init, xbounds, tbounds)
        sol_u = Upwind(k, h, init, xbounds, tbounds)


        tempu = abs(analytical_Traffic(x, t[Nt-1]) - sol_u[:,Nt-1])
        templf = abs(analytical_Traffic(x, t[Nt-1]) - sol_lf[:, Nt-1])
        tempr = abs(analytical_Traffic(x, t[Nt-1]) - sol_R[:, Nt-1])

        maxerror_u[hs, 0] = h * sum(tempu)
        maxerror_u[hs, 1] = pow(h * sum(pow(tempu, 2)), 1/2)
        maxerror_u[hs, 2] = max(tempu)

        maxerror_lf[hs, 0] = h * sum(templf)
        maxerror_lf[hs, 1] = pow(h *sum(pow(templf, 2)), 1 / 2)
        maxerror_lf[hs, 2] = max(templf)

        maxerror_R[hs, 0] = h * sum(tempr)
        maxerror_R[hs, 1] = pow(h * sum(pow(tempr, 2)), 1 / 2)
        maxerror_R[hs, 2] = max(tempr)

        print("\nMax Error when h = %f" % hvals[hs])
        print("Upwind Errors:")
        for i in (0, 1):
            print("\t e_%d = %f" % (i + 1, maxerror_u[hs, i]))
        print("\t e_inf = %f" % maxerror_u[hs, 2])

        print("Lax Friedrichs Errors:")
        for i in (0, 1):
            print("\t e_%d = %f" % (i + 1, maxerror_lf[hs, i]))
        print("\t e_inf = %f" % maxerror_lf[hs, 2])

        print("Ritchmeyer Errors:")
        for i in (0, 1):
            print("\t e_%d = %f" % (i + 1, maxerror_R[hs, i]))
        print("\t e_inf = %f" % maxerror_R[hs, 2])

    logH = np.log(hvals)

    figure, axis = plt.subplots(2, 2)

    for i in (0, 1, 2):
        figplace1 = (0, 0, 1, 1)
        figplace2 = (0, 1, 0, 1)

        slope_u = (-np.log(maxerror_u[len(hvals) - 1, i]) + np.log(maxerror_u[len(hvals) - 2, i])) / (
                -logH[len(hvals) - 1] + logH[len(hvals) - 2])
        slope_lf = (-np.log(maxerror_lf[len(hvals) - 1, i]) + np.log(maxerror_lf[len(hvals) - 2, i])) / (
                -logH[len(hvals) - 1] + logH[len(hvals) - 2])
        slope_R = (-np.log(maxerror_R[len(hvals) - 1, i]) + np.log(maxerror_R[len(hvals) - 2, i])) / (
                -logH[len(hvals) - 1] + logH[len(hvals) - 2])

        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_u[:, i]),
                                                  label="Up_Error - Slope %f" % slope_u)
        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_lf[:, i]),
                                                  label="LF_Error - Slope %f" % slope_lf)
        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_R[:, i]),
                                                 label="R_Error - Slope %f" % slope_R)

        axis[figplace1[i], figplace2[i]].legend()
        if i != 2:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
        else:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

    plt.show()