import numpy as np
import matplotlib.pyplot as plt
from docutils.nodes import label
from numpy import where

def numerical_solver(A, init, h, k, xbound, tbound, name):

    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)

    sol = np.zeros((Nx, Nt))

    for t in range(0, Nt):
        sol[:, t] = y

        if name == "LF":
            y[1:-1] = 1 / 2 * (y[:-2] + y[2:]) - k / (2 * h) * A * (y[2:] - y[:-2])

        elif name == "LW":
            y[1:-1] = y[1:-1] - (k / (2 * h)) * A * (y[2:] - y[:-2]) + (k * k / (2 * h * h)) * A * A * (
                        y[2:] - 2 * y[1:-1] + y[:-2])
        else:
            y[1:-1] = y[1:-1] - (k / h) * A * (y[1:-1] - y[:-2])

    return sol


def initial_condition1(x):
    y = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        if(x[i] >= 0):
            y[i] = 1
        elif(x[i] < 0):
            y[i] = 0
    return(y)

def analytical1(x, t, init):
    t1 = init(x - t)
    return t1


def initial_condition2(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f


if __name__ == '__main__':
    system = 1
    init = initial_condition2
    PLOT = 0

    h = 0.01
    k = h/3

    xbounds = (-1, 2)
    tbounds = (0, 0.5)

    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1

    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

    A = 1

    sol_lw = numerical_solver(A, init, h, k, xbounds, tbounds, "LW")
    sol_lf = numerical_solver(A, init, h, k, xbounds, tbounds, "LF")
    sol_u = numerical_solver(A, init, h, k, xbounds, tbounds, "U")

    Ana = analytical1(x, 0, init)

    if PLOT:
        ### PLOTTING Solutions
        plt.ion()
        figure, axis = plt.subplots(1)

        line0, = axis.plot(x, Ana, color='red', label='Analytical Solution')

        line1, = axis.plot(x, sol_lw[:, 0], color='blue', label="L-W")

        line2, = axis.plot(x, sol_lf[:, 0], color='green', label="L-F")

        line3, = axis.plot(x, sol_u[:, 0], color='orange', label="Upwind")

        plt.legend()

        for i in range(0, Nt):
            line0.set_ydata(analytical1(x, t[i], init))
            line1.set_ydata(sol_lw[:, i])
            line2.set_ydata(sol_lf[:, i])
            line3.set_ydata(sol_u[:, i])

            figure.canvas.draw()
            figure.canvas.flush_events()


    else:
        ## Calculating and Plotting Error
        hvals = (.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16, 0.1/32, 0.1/64, 0.1/128, 0.1/256)
        hs = -1

        maxerror_u = np.zeros((len(hvals), 3))
        maxerror_lw = np.zeros((len(hvals), 3))
        maxerror_lf = np.zeros((len(hvals), 3))

        err_u = np.zeros((len(hvals), 3))
        err_lf = np.zeros((len(hvals), 3))
        err_lw = np.zeros((len(hvals), 3))

        for h in hvals:
            hs = hs+1
            k = h / 2

            Nx = int((xbounds[1] - xbounds[0]) / h) + 1
            Nt = int((tbounds[1] - tbounds[0]) / k) + 1
            x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
            t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time


            sol_u = numerical_solver(A, init, h, k, xbounds, tbounds, "UP")[:, Nt-1]
            sol_lf = numerical_solver(A, init, h, k, xbounds, tbounds, "LF")[:, Nt-1]
            sol_lw = numerical_solver(A, init, h, k, xbounds, tbounds, "LW")[:, Nt-1]

            analytical = analytical1(x, t[Nt-1], init)

            temp = abs(analytical - sol_u)
            maxerror_u[hs, 0] = h * sum(temp)
            maxerror_u[hs, 1] = pow(h * sum((pow(temp, 2))), 1/2)
            maxerror_u[hs, 2] = max(temp)

            err_u[hs] = (sum(temp), sum(pow(temp, 2)), max(temp))

            temp = abs(analytical - sol_lw)
            maxerror_lw[hs, 0] = h * sum(temp)
            maxerror_lw[hs, 1] = pow(h * sum((pow(temp, 2))), 1 / 2)
            maxerror_lw[hs, 2] = max(temp)

            err_lw[hs] = (sum(temp), sum(pow(temp, 2)), max(temp))

            temp = abs(analytical - sol_lf)
            maxerror_lf[hs, 0] = h * sum(temp)
            maxerror_lf[hs, 1] = pow(h * sum((pow(temp, 2))), 1 / 2)
            maxerror_lf[hs, 2] = max(temp)

            err_lf[hs] = (sum(temp), sum(pow(temp, 2)), max(temp))

            print("\nMax Error when h = %f" % hvals[hs])
            print("Upwind Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_u[hs, i]))
            print("\t e_inf = %f" % maxerror_u[hs, 2])
    
            print("Lax Friedrichs Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_lf[hs, i]))
            print("\t e_inf = %f" % maxerror_lf[hs, 2])
    
            print("Lax Wendroff Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_lw[hs, i]))
            print("\t e_inf = %f" % maxerror_lw[hs, 2])

        logH = np.log(hvals)

        figure, axis = plt.subplots(2, 2)
        if init == initial_condition1:
            figure.suptitle("System %d, with discontinuous initial condition" % system)
        else:
            figure.suptitle("System %d, with smooth initial condition" % system)

        print(err_u)
        print("\n")
        print(err_lf)
        print("\n")
        print(err_lw)

        for i in (0, 1, 2):
            figplace1 = (0, 0, 1, 1)
            figplace2 = (0, 1, 0, 1)

            slope_u = (-np.log(maxerror_u[len(hvals) - 1, i]) + np.log(maxerror_u[len(hvals) - 2, i])) / (
                        -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            slope_lf = (-np.log(maxerror_lf[len(hvals) - 1, i]) + np.log(maxerror_lf[len(hvals) - 2, i])) / (
                        -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            slope_lw = (-np.log(maxerror_lw[len(hvals) - 1, i]) + np.log(maxerror_lw[len(hvals) - 2, i])) / (
                        -logH[len(hvals) - 1] + logH[len(hvals) - 2])

            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_u[:, i]), label="Up_Error - Slope %f" % slope_u)
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_lf[:, i]), label="LF_Error - Slope %f" % slope_lf)
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_lw[:, i]), label="LW_Error - Slope %f" % slope_lw)

            axis[figplace1[i], figplace2[i]].legend()
            if i != 2:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
            else:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

        plt.show()

        figure, axis = plt.subplots(2, 2)

        axis[0,0].plot(-logH, -np.log(err_u[:, 0]), label="Upwind")
        axis[0,0].plot(-logH, -np.log(err_lf[:, 0]), label="LF")
        axis[0,0].plot(-logH, -np.log(err_lw[:, 0]), label="LW")
        axis[0,0].legend()
        axis[0,0].set_title("Error")

        axis[0, 1].plot(-logH, -np.log(err_u[:, 1]), label="Upwind")
        axis[0, 1].plot(-logH, -np.log(err_lf[:, 1]), label="LF")
        axis[0, 1].plot(-logH, -np.log(err_lw[:, 1]), label="LW")
        axis[0, 1].set_title("Error Squared")

        axis[1, 0].plot(-logH, -np.log(err_u[:, 2]), label="Upwind")
        axis[1, 0].plot(-logH, -np.log(err_lf[:, 2]), label="LF")
        axis[1, 0].plot(-logH, -np.log(err_lw[:, 2]), label="LW")
        axis[1, 0].set_title("Max Error")

        plt.show()