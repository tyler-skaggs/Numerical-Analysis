import numpy as np
import matplotlib.pyplot as plt
from numpy import where
import scipy.optimize as opt
from Solvers import solver


def traffic(x):
    umax = 1
    pmax = 2
    return x * umax * (1 - x / pmax)

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def linear(x):
    return a*x

def linear_prime(x):
    return a

def BL(x):
    return pow(x,2) / (pow(x,2) + a * pow(1 - x,2))

def BL_prime(x):
    return -1*(2*a * (x - 1) * x) / pow( (a+1) * pow(x,2) - 2 * a * x + a, 2)

## Analytical Solutions and Initial Conditions
def analytical1(x, t):
    y = np.zeros(np.size(x))
    ul = 0
    ur = 1
    for i in range(0, np.size(x)):
        if x[i] <= ul * t:
            y[i] = ul
        elif x[i] > t * ur:
            y[i] = ur
        else:
            y[i] = x[i] / t
    return y

def analytical2(x, t):
    y = np.zeros(np.size(x))
    ul = 1
    ur = 0
    s = (ul + ur)/2

    for i in range(0, np.size(x)):
        if x[i] <= s * t:
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

def analytic_GreenLight(x, t):
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

def analytic_linear(x, t):
    y = np.zeros(np.size(x))
    ul = 1
    ur = 0
    for i in range(0, len(x)):
        if x[i] < a*t:
            y[i] = ul
        else:
            y[i] = ur
    return y

def initial_smooth(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f

def analytic_linear_smooth(x, t):
    return initial_smooth(x - a * t)

def initial_sin(x):
    y = np.zeros(np.size(x))
    for i in range(0,len(x)):
        if x[i] == 0 or x[i] ==1:
            y[i] = 0
        else:
            y[i] = np.sin(2 * np.pi * x[i])

    return y

def analyticalBL(x, t):
    y = np.zeros(np.size(x))
    ul = 1
    ur = 0

    s = (BL(ur) - BL(ul))/(ur - ul)

    x_crit = t * BL_prime(pow(a / (1 + a), 1/2))

    for i in range(0, np.size(x)):
        if ul < pow(a / (1 + a), 1/2):
            if x[i] <= s*t:
                y[i] = ul
            else:
                y[i] = ur
        else:
            if x[i] <= BL_prime(ul)*t:
                y[i] = ul

            elif x[i] >= x_crit:
                y[i] = ur

            else:
                def temp(u):
                    return t*BL_prime(u) - x[i]
                y[i] = opt.brentq(temp, a, ul)

    return y

if __name__ == '__main__':
    global a
    a = 0.5

    analytic = analytic_linear_smooth
    def init(x):
        return analytic(x, 0)

    #init = initial_sin
    problem = linear#burgers
    deriv = linear_prime#burgers_prime
    plot = 1

    h = 0.00000000000000001
    k = h / 3
    xbounds = (0.5 - h*100000, 0.5+h*100000)
    tbounds = (0, h*10)
    Nx = int((xbounds[1] - xbounds[0]) / h) + 1
    Nt = int((tbounds[1] - tbounds[0]) / k) + 1
    x = np.linspace(xbounds[0], xbounds[1], Nx)  # discretization of space
    t = np.linspace(tbounds[0], tbounds[1], Nt)  # discretization of time

    if plot == 1:
        solLF = solver(k, h, init, xbounds, tbounds, problem, deriv, "LF")
        #solR = solver(k, h, init, xbounds, tbounds, problem, deriv, "R")
        #solM = solver(k, h, init, xbounds, tbounds, problem, deriv, "M")
        solU = solver(k, h, init, xbounds, tbounds, problem, deriv, "U")
        #solG = solver(k, h, init, xbounds, tbounds, problem, deriv, "G")
        solLW = solver(k, h, init, xbounds, tbounds, problem, deriv, "LW")

        plt.ion()
        figure = plt.figure()
        axis = figure.add_subplot(111)

        line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')  # Returns a tuple of line objects, thus the comma
        line1, = axis.plot(x, solLF[:, 0], 'blue', label='Lax-Friedrichs')
        #line2, = axis.plot(x, solR[:, 0], 'green', label='Richtmyer')
        #line3, = axis.plot(x, solM[:, 0], 'orange', label='MacCormack')
        #line4, = axis.plot(x, solU[:, 0], 'yellow', label='Upwind')
        #line5, = axis.plot(x, solG[:, 0], 'cyan', label='Gudunov')
        #line6, = axis.plot(x, solLW[:, 0], 'green', label='Lax-Wendroff')

        plt.legend()
        plt.xlabel("x")
        if problem == burgers:
            plt.title(r"Burger's Equations")
            plt.ylabel(r"$u(x,t)$")
        elif problem == linear:
            plt.title(r"Linear Equations")
            plt.ylabel(r"$u(x,t)$")
        else:
            plt.title("Traffic Flow Problem")
            plt.ylabel(r"$\rho(x,t)$")

        plt.ylim(min(solU[:, 0]) - 0.2, max(solU[:,0]) + 0.2)

        text = plt.text(0, 0, "t = 0")

        for i in range(1, Nt):
            text.set_text("t = %f" % t[i])
            line0.set_ydata(analytic(x, t[i]))
            line1.set_ydata(solLF[:, i])
            #line2.set_ydata(solR[:, i])
            #line3.set_ydata(solM[:, i])
            #line4.set_ydata(solU[:, i])
            #line5.set_ydata(solG[:, i])
            #line6.set_ydata(solLW[:, i])
            figure.canvas.draw()
            figure.canvas.flush_events()

    ## Calculating and Plotting Error
    else:
        hvals = (.1, 0.1/2, 0.1/4, 0.1/8, 0.1/16)#, 0.1/32, 0.1/64, 0.1/128, 0.1/256)
        hs = -1

        maxerror_u = np.zeros((len(hvals), 3))
        maxerror_lf = np.zeros((len(hvals), 3))
        maxerror_R = np.zeros((len(hvals), 3))
        maxerror_M = np.zeros((len(hvals), 3))
        maxerror_G = np.zeros((len(hvals), 3))

        for h in hvals:
            hs = hs + 1
            k = h / 2

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

            sol_lf = solver(k, h, init, xbounds, tbounds, problem, deriv, "LF")
            sol_R = solver(k, h, init, xbounds, tbounds, problem, deriv, "R")
            sol_M = solver(k, h, init, xbounds, tbounds, problem, deriv, "M")
            sol_u = solver(k, h, init, xbounds, tbounds, problem, deriv, "U")
            sol_G = solver(k, h, init, xbounds, tbounds, problem, deriv, "G")

            tempu = abs(analytic(x, t[Nt-1]) - sol_u[:,Nt-1])
            templf = abs(analytic(x, t[Nt-1]) - sol_lf[:, Nt-1])
            tempr = abs(analytic(x, t[Nt-1]) - sol_R[:, Nt-1])
            tempm = abs(analytic(x, t[Nt - 1]) - sol_M[:, Nt - 1])
            tempG = abs(analytic(x, t[Nt - 1]) - sol_G[:, Nt - 1])

            maxerror_u[hs, 0] = h * sum(tempu)
            maxerror_u[hs, 1] = pow(h * sum(pow(tempu, 2)), 1/2)
            maxerror_u[hs, 2] = max(tempu)

            maxerror_lf[hs, 0] = h * sum(templf)
            maxerror_lf[hs, 1] = pow(h * sum(pow(templf, 2)), 1 / 2)
            maxerror_lf[hs, 2] = max(templf)

            maxerror_R[hs, 0] = h * sum(tempr)
            maxerror_R[hs, 1] = pow(h * sum(pow(tempr, 2)), 1 / 2)
            maxerror_R[hs, 2] = max(tempr)

            maxerror_M[hs, 0] = h * sum(tempm)
            maxerror_M[hs, 1] = pow(h * sum(pow(tempm, 2)), 1 / 2)
            maxerror_M[hs, 2] = max(tempm)

            maxerror_G[hs, 0] = h * sum(tempG)
            maxerror_G[hs, 1] = pow(h * sum(pow(tempG, 2)), 1 / 2)
            maxerror_G[hs, 2] = max(tempG)

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

            print("MacCormack Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_M[hs, i]))
            print("\t e_inf = %f" % maxerror_M[hs, 2])

            print("Gudunov Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_G[hs, i]))
            print("\t e_inf = %f" % maxerror_G[hs, 2])

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
            slope_M = (-np.log(maxerror_M[len(hvals) - 1, i]) + np.log(maxerror_M[len(hvals) - 2, i])) / (
                    -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            slope_G = (-np.log(maxerror_G[len(hvals) - 1, i]) + np.log(maxerror_G[len(hvals) - 2, i])) / (
                    -logH[len(hvals) - 1] + logH[len(hvals) - 2])

            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_u[:, i]),
                                                      label="Up_Error - Slope %f" % slope_u, color="yellow")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_lf[:, i]),
                                                      label="LF_Error - Slope %f" % slope_lf, color="blue")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_R[:, i]),
                                                     label="R_Error - Slope %f" % slope_R, color="green")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_M[:, i]),
                                                  label="M_Error - Slope %f" % slope_M, color="orange")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log(maxerror_G[:, i]),
                                                  label="G_Error - Slope %f" % slope_G, color="red")

            axis[figplace1[i], figplace2[i]].legend()
            if i != 2:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
            else:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

        plt.show()