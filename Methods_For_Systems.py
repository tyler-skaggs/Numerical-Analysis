import numpy as np
import matplotlib.pyplot as plt
from numpy import where


def numerical_solver(A, Ap, An, init, h, k, xbound, tbound, name):

    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = np.matrix([init(x), init(x)])

    sol1 = np.zeros((Nx, Nt))
    sol2 = np.zeros((Nx, Nt))

    for t in range(0, Nt):
        sol1[:, t] = y[0]
        sol2[:, t] = y[1]

        if name == "LF":
            y[:, 1:-1] = 1 / 2 * (y[:, :-2] + y[:, 2:]) - k / (2 * h) * A * (y[:, 2:] - y[:, :-2])

        elif name == "LW":
            y[:, 1:-1] = y[:, 1:-1] - (k / (2 * h)) * A * (y[:, 2:] - y[:, :-2]) + (k * k / (2 * h * h)) * A * A * (
                        y[:, 2:] - 2 * y[:, 1:-1] + y[:, :-2])
        else:
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
    x_left = 2.75
    x_right = 3.25
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
    system = 2
    init = initial_condition2

    h = 0.01
    k = h/3
    xbounds = (0, 6)
    tbounds = (0, 0.5)
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

    solA, solB = numerical_solver(A2, A2p, A2n, init, h, k, xbounds, tbounds, "LW")

    AnaA, AnaB = analytical1(x, 0, init)

    """
    ### PLOTTING Solutions
    plt.ion()
    figure, axis = plt.subplots(2)

    #line0, = axis[0].plot(x, AnaA, color='red', label='Analytical Solution')
    #line1, = axis[1].plot(x, AnaB, color='red', label='Analytical Solution')

    line2, = axis[0].plot(x, solA[:, 0], color='blue', label="Numerical Solution")  # Returns a tuple of line objects, thus the comma
    axis[0].title.set_text('Equation A')

    line3, = axis[1].plot(x, solB[:, 0], color='blue', label="Numerical Solution")  # Returns a tuple of line objects, thus the comma
    axis[1].title.set_text('Equation B')

    plt.legend()

    for i in range(0, Nt):
        #line0.set_ydata(analytical2(x, t[i], init)[0])
        #line1.set_ydata(analytical2(x, t[i], init)[1])
        line2.set_ydata(solA[:, i])
        line3.set_ydata(solB[:, i])

        figure.canvas.draw()
        figure.canvas.flush_events()

    """

    ## Calculating and Plotting Error
    hvals = (.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16, 0.1/32, 0.1/64, 0.1/128, 0.1/256)
    hs = -1

    maxerror_u = np.zeros((len(hvals), 3))
    maxerror_lw = np.zeros((len(hvals), 3))
    maxerror_lf = np.zeros((len(hvals), 3))

    for h in hvals:
        hs = hs+1
        k = h / 3

        Nx = int((xbounds[1] - xbounds[0]) / h) + 1
        Nt = int((tbounds[1] - tbounds[0]) / k) + 1
        x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
        t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

        if system == 1:
            solA_u, solB_u = numerical_solver(A1, A1p, A1n, init, h, k, xbounds, tbounds, "UP")
            solA_lf, solB_lf = numerical_solver(A1, A1p, A1n, init, h, k, xbounds, tbounds, "LF")
            solA_lw, solB_lw = numerical_solver(A1, A1p, A1n, init, h, k, xbounds, tbounds, "LW")

        if system == 2:
            solA_u, solB_u = numerical_solver(A2, A2p, A2n, init, h, k, xbounds, tbounds, "UP")
            solA_lf, solB_lf = numerical_solver(A2, A2p, A2n, init, h, k, xbounds, tbounds, "LF")
            solA_lw, solB_lw = numerical_solver(A2, A2p, A2n, init, h, k, xbounds, tbounds, "LW")


        sol_u = np.array((solA_u[:, Nt-1], solB_u[:, Nt-1]))
        sol_lw = np.array((solA_lw[:, Nt-1], solB_lw[:, Nt-1]))
        sol_lf = np.array((solA_lf[:, Nt-1], solB_lf[:, Nt-1]))

        if system == 1:
            AnaA = analytical1(x, t[Nt-1], init)[0]
            AnaB = analytical1(x, t[Nt-1], init)[1]

        if system == 2:
            AnaA = analytical2(x, t[Nt-1], init)[0]
            AnaB = analytical2(x, t[Nt-1], init)[1]

        analytical = np.array((AnaA, AnaB))

        temperr_u = np.zeros(len(x))
        temperr_lw = np.zeros(len(x))
        temperr_lf = np.zeros(len(x))

        error_1_u = 0
        error_2_u = 0
        error_1_lw = 0
        error_2_lw = 0
        error_1_lf = 0
        error_2_lf = 0

        for j in range(1, Nx):

            temp = abs(analytical[:, j] - sol_u[:, j])
            error_1_u = error_1_u + sum(temp)
            error_2_u = error_2_u + (pow(temp[0], 2) + pow(temp[1], 2))
            temperr_u[j] = max(abs(temp))

            temp = abs(analytical[:, j] - sol_lw[:, j])
            error_1_lw = error_1_lw + sum(temp)
            error_2_lw = error_2_lw + (pow(temp[0], 2) + pow(temp[1], 2))
            temperr_lw[j] = max(abs(temp))

            temp = abs(analytical[:, j] - sol_lf[:, j])
            error_1_lf = error_1_lf + sum(temp)
            error_2_lf = error_2_lf + (pow(temp[0], 2) + pow(temp[1], 2))
            temperr_lf[j] = max(abs(temp))


        ## Finding Maximum of Error

        maxerror_u[hs, 0] = h * error_1_u
        maxerror_u[hs, 1] = pow(h * error_2_u, 1/2)
        maxerror_u[hs, 2] = max(temperr_u)

        maxerror_lw[hs, 0] = h * error_1_lw
        maxerror_lw[hs, 1] = pow(h * error_2_lw, 1/2)
        maxerror_lw[hs, 2] = max(temperr_lw)

        maxerror_lf[hs, 0] = h * error_1_lf
        maxerror_lf[hs, 1] = pow(h * error_2_lf, 1/2)
        maxerror_lf[hs, 2] = max(temperr_lf)

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



