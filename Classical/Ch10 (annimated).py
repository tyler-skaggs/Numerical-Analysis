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


def initial_condition3(x):
    return pow(np.sin(x), 2)


if __name__ == '__main__':
    A = 1
    h = 0.0001
    k = h / 2

    tmin, tmax = 0.0, 1.0  # start and stop time of simulation
    xmin, xmax = 0, 2  # start and end of spatial domain

    solvers = [lax_friedrichs]

    Nx = int((xmax - xmin) / h) + 1
    Nt = int((tmax - tmin) / k) + 1

    x = np.linspace(xmin, xmax, Nx)  # discretization of space
    t = np.linspace(tmin, tmax, Nt)  # discretization of time

    solutions = np.zeros((len(solvers), len(t), len(x)))
    analytical = np.zeros((len(t), len(x)))  # holds the analytical solution

    for j, solver in enumerate(solvers):  # Solve for all solvers in list
        u = initial_condition2(x)
        un = np.zeros((np.size(t), np.size(x)))  # holds the numerical solution

        for i, tt in enumerate(t[1:]):

            if j == 0:
                analytical[i, :] = initial_condition2(x - A * tt)  # compute analytical solution for this time step

            u_bc = interpolate.interp1d(x[-2:], u[-2:])  # interplate at right bndry

            u[1:-1] = solver(u[:], A, h, k)  # calculate numerical solution of interior
            u[-1] = u_bc(x[-1] - A * k)  # interpolate along a characteristic to find the boundary value

            un[i, :] = u[:]  # storing the solution for plotting

        solutions[j, :, :] = un

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(xmin, xmax), ylim=(np.min(un), np.max(un) * 1.1))
    plt.ylim(-1.25, 1.25)

    lines = []  # list for plot lines for solvers and analytical solutions
    legends = []  # list for legends for solvers and analytical solutions

    for solver in solvers:
        line, = ax.plot([], [])
        lines.append(line)
        legends.append(solver.__name__)

    line, = ax.plot([], [])  # add extra plot line for analytical solution
    lines.append(line)
    legends.append('Analytical')

    plt.xlabel('x-coordinate [-]')
    plt.ylabel('Amplitude [-]')
    plt.legend(legends, loc=3, frameon=False)


    # initialization function: plot the background of each frame
    def init():
        for line in lines:
            line.set_data([], [])
        return lines,


    # animation function.  This is called sequentially
    def animate(i):
        for k, line in enumerate(lines):
            if (k == 0):
                line.set_data(x, un[i, :])
            else:
                line.set_data(x, analytical[i, :])
        return lines,


    def animate_alt(i):
        for k, line in enumerate(lines):
            if (k == len(lines) - 1):
                line.set_data(x, analytical[i, :])
            else:
                line.set_data(x, solutions[k, i, :])

        i += 1000
        return lines,


    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate_alt, init_func=init, frames=int(tmax/k), interval=100, blit=False)

    plt.show()