import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy import where
from matplotlib import animation


def backward_euler(y, A, h, k):
    y[1:-1] = y[1:-1] - (k / (2 * h)) * A * (y[2:] - y[:-2])
    return(y[1:-1])


def one_sided_left(y, A, h, k):
    y[1:-1] = y[1:-1] - (k / h) * A * (y[1:-1] - y[:-2])
    return(y[1:-1])


def one_sided_right(y, A, h, k):
    y[1:-1] = y[1:-1] - (k / h) * A * (y[2:] - y[1:-1])
    return(y[1:-1])


def lax_friedrichs(y, A, h, k):
    y[1:-1] = 1 / 2 * (y[:-2] + y[2:]) - k / (2*h) * A * (y[2:] - y[:-2])
    return(y[1:-1])


def leapfrog(A, init, h, k, N, xbound):
    if(h < 2*k):
        k = h / 2

    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    yt = y
    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        prev = yt

        if(step == 0):
            y[1:-1] = y[1:-1]
        else:
            y[1:-1] = prev[1:-1] - k / (2 * h) * A * (y[2:] - y[:-2])

        yt = y


def lax_wendroff(y, A, h, k):
    y[1:-1] = y[1:-1] - (k / (2 * h)) * A * (y[2:] - y[:-2]) + (k * k / (2 * h * h)) * A * A * (y[2:] - 2 * y[1:-1] + y[:-2])
    return(y[1:-1])


def beam_warming(y, A, h, k):
    y[2:] = y[2:] - k / (2 * h) * A * (3 * y[2:] - 4 * y[1:-1] + y[:-2]) + k * k / (2 * h * h) * A * A * (y[2:] - 2 * y[1:-1] + y[:-2])
    return(y[2:])


def initial_condition1(x):
    y = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        if(x[i] >= 0.5):
            y[i] = 1
        elif(x[i] < 0.5):
            y[i] = 0
    return(y)


def initial_condition2(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f


if __name__ == '__main__':
    A = 1
    init = initial_condition2

    hvals = (.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16, 0.1/32, 0.1/64, 0.1/128, 0.1/256)
    maxerror = np.zeros((len(hvals), 3, 4))

    hs = -1
    for h in hvals:
        hs = hs + 1
        k = h / 2

        tmin, tmax = 0, 1  # start and stop time of simulation
        xmin, xmax = 0, 4  # start and end of spatial domain

        solvers = [lax_wendroff, lax_friedrichs, one_sided_left]

        Nx = int((xmax - xmin) / h) + 1
        Nt = int((tmax - tmin) / k) + 1

        x = np.linspace(xmin, xmax, Nx)  # discretization of space
        t = np.linspace(tmin, tmax, Nt)  # discretization of time

        solutions = np.zeros((len(solvers), len(t), len(x)))
        analytical = np.zeros((len(t), len(x)))  # holds the analytical solution

        for j, solver in enumerate(solvers):  # Solve for all solvers in list
            u = init(x)
            un = np.zeros((np.size(t), np.size(x)))  # holds the numerical solution

            for i, tt in enumerate(t[1:]):

                if j == 0:
                    analytical[i, :] = init(x - A * tt)  # compute analytical solution for this time step

                u_bc = interpolate.interp1d(x[-2:], u[-2:])  # interplate at right bndry

                u[1:-1] = solver(u[:], A, h, k)  # calculate numerical solution of interior
                u[-1] = u_bc(x[-1] - A * k)  # interpolate along a characteristic to find the boundary value

                un[i, :] = u[:]  # storing the solution for plotting

            solutions[j, :, :] = un

        error_1 = np.zeros((len(solvers)))
        error_2 = np.zeros((len(solvers)))
        error_3 = np.zeros((len(solvers)))
        error_inf = np.zeros((len(solvers)))

        for j in range(0, len(solvers)):
            temp = abs(analytical[Nt-2, :] - solutions[j, Nt-2, :])
            error_1[j] = h*sum(temp)
            error_2[j] = pow(h*sum(pow(temp, 2)), 1/2)
            error_3[j] = pow(h*sum(pow(temp, 3)), 1/3)
            error_inf[j] = max(temp)

        for k in (0, 1, 2):
            maxerror[hs, k, 0] = error_1[k]
            maxerror[hs, k, 1] = error_2[k]
            maxerror[hs, k, 2] = error_3[k]
            maxerror[hs, k, 3] = error_inf[k]

    for k in (0, 1, 2, 3):
        print("\nMax Error when h = %f" % hvals[k])
        print("Lax-Wendroff Errors:")
        for i in (0, 1, 2):
            print("\t e_%d = %f" % (i+1, maxerror[k, 0, i]))
        print("\t e_inf = %f" % maxerror[k, 0, 3])

        print("Lax-Friedrichs Errors:")
        for i in (0, 1, 2):
            print("\t e_%d = %f" % (i+1, maxerror[k, 1, i]))
        print("\t e_inf = %f" % maxerror[k, 1, 3])

        print("Upwind Errors:")
        for i in (0, 1, 2):
            print("\t e_%d = %f" % (i+1, maxerror[k, 2, i]))
        print("\t e_inf = %f" % maxerror[k, 2, 3])

    LWer = maxerror[:, 0, :]
    LFer = maxerror[:, 1, :]
    UWer = maxerror[:, 2, :]

    logH = np.log(hvals)

    figure, axis = plt.subplots(2, 2)
    for i in (0, 1, 2, 3):
        figplace1 = (0, 0, 1, 1)
        figplace2 = (0, 1, 0, 1)
        LWslope = (-np.log(LWer[len(hvals)-1, i]) + np.log(LWer[len(hvals)-2, i])) / (-logH[len(hvals)-1] + logH[len(hvals)-2])
        LFslope = (-np.log(LFer[len(hvals)-1, i]) + np.log(LFer[len(hvals)-2, i])) / (-logH[len(hvals)-1] + logH[len(hvals)-2])
        UWslope = (-np.log(UWer[len(hvals)-1, i]) + np.log(UWer[len(hvals)-2, i])) / (-logH[len(hvals)-1] + logH[len(hvals)-2])

        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(LWer[:, i]), label="Lax-Wendroff Error - Slope %f" % LWslope)
        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(LFer[:, i]), label="Lax-Fredric Error - Slope %f" % LFslope)
        axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(UWer[:, i]), label="Upwind Error - Slope %f" % UWslope)
        axis[figplace1[i], figplace2[i]].legend()
        if i != 3:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i+1))
        else:
            axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

    plt.show()

"""figure, axis = plt.subplots(2, 6)
        figure.suptitle("Error for Lax-Wendroff (left), Lax-Friedrichs (middle), Upwind (right)\n h = %f" %h)

        axis[0, 0].plot(t, error_1[0, :])
        axis[0, 0].set_title(r"$e_1$")

        axis[0, 1].plot(t, error_2[0, :])
        axis[0, 1].set_title(r"$e_2$")

        axis[1, 0].plot(t, error_3[0, :])
        axis[1, 0].set_title(r"$e_3$")

        axis[1, 1].plot(t, error_inf[0, :])
        axis[1, 1].set_title(r"$e_\infty$")

        axis[0, 2].plot(t, error_1[1, :])
        axis[0, 2].set_title(r"$e_1$")

        axis[0, 3].plot(t, error_2[1, :])
        axis[0, 3].set_title(r"$e_2$")

        axis[1, 2].plot(t, error_3[1, :])
        axis[1, 2].set_title(r"$e_3$")

        axis[1, 3].plot(t, error_inf[1, :])
        axis[1, 3].set_title(r"$e_\infty$")

        axis[0, 4].plot(t, error_1[2, :])
        axis[0, 4].set_title(r"$e_1$")

        axis[0, 5].plot(t, error_2[2, :])
        axis[0, 5].set_title(r"$e_2$")

        axis[1, 4].plot(t, error_3[2, :])
        axis[1, 4].set_title(r"$e_3$")

        axis[1, 5].plot(t, error_inf[2, :])
        axis[1, 5].set_title(r"$e_\infty$")

        plt.show()"""