import numpy as np

from Solvers import solver
import matplotlib.pyplot as plt
from numpy import where
from scipy.stats import linregress
from ENO import *

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def linear_deriv(x):
    if type(x) == type(1.0):
        return 1
    else:
        return np.ones(len(x))

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

def initial_smooth(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f

def analytic_linear_smooth(x, t):
    return initial_smooth(x - t)

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

def traffic(x):
    umax = 1
    pmax = 2
    return x * umax * (1 - x / pmax)

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

def initial_smooth(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f

if __name__ == '__main__':
    def init(x):
        return np.sin(np.pi * x)
        #return analytic(x,0)

    def analytic(x,t):
        return init(x - t)


    problem = burgers_prime
    deriv = linear_deriv

    dx = 1/50
    dt = dx/2

    a = -1
    b = 1
    time = 1

    Nx = int((b-a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a,b,Nx)
    t = np.linspace(0, time, Nt)

    plot = 0

    if plot == 1:
        #solLW = solver(dt, dx, init, (a,b), (0,time), problem, deriv, "CD")

        eno = ENO(l = a, r = b, dx=dx, dt=dt, init = init, problem = problem, deriv=deriv)
        eno.set_initial()

        eeno = EENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv)
        eeno.set_initial()

        weno = WENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv, rr = 2)
        weno.set_initial()

        eweno = EWENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv, rr=3)
        eweno.set_initial()

        plt.ion()
        figure = plt.figure()
        axis = figure.add_subplot(111)

        line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')  # Returns a tuple of line objects, thus the comma
        line1, = axis.plot(eno.xc[2:-2], eno.u[2:-2], color = 'green', label='ENO Solution')  # Returns a tuple of line objects, thus the comma
        line2, = axis.plot(eeno.xc[5:-5], eeno.u[5:-5], color='blue', label='EENO Solution')  # Returns a tuple of line objects, thus the comma
        lineWENO, = axis.plot(eno.xc[2:-2], weno.u[3:-3], color='black', label='WENO2 Solution')
        lineEWENO, = axis.plot(eweno.xc[3:-3], eweno.u[3:-3], color='purple', label='E-WENO Solution')
        #lineLW, = axis.plot(x, init(x), color='black', label='CD Solution')  # Returns a tuple of line objects, thus the comma

        plt.ylim(-1.5, 1.5)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x,t)")

        text = plt.text(0, 0, "t = 0")

        t = 0
        i = 0
        while t < time-dt/2:
            t += dt
            eno.Runge_Kutta()
            eeno.Runge_Kutta()
            weno.Runge_Kutta()
            eweno.Runge_Kutta()

            text.set_text("t = %f" % t)

            line0.set_ydata(analytic(x, t))
            line1.set_ydata(eno.u[2:-2])
            line2.set_ydata(eeno.u[5:-5])
            lineWENO.set_ydata(weno.u[3:-3])
            lineEWENO.set_ydata(eweno.u[3:-3])
            #lineLW.set_ydata(solLW[:, i+1])

            figure.canvas.draw()
            figure.canvas.flush_events()

            i+=1
        plt.ioff()
        plt.show()

    else:
        Nvals = np.array([10, 20, 40])#, 80])#, 160])#, 320])#, 640, 1280])
        hvals = (b - a)/(Nvals - 1) #(.1, 0.1 / 2, 0.1 / 4, 0.1 / 8, 0.1 / 16, 0.1 / 32)#, 0.1 / 64, 0.1 / 128, 0.1 / 256)
        hs = -1

        maxerror_ENO = np.zeros((len(hvals), 3))
        maxerror_EENO = np.zeros((len(hvals), 3))
        maxerror_WENO = np.zeros((len(hvals), 3))
        maxerror_EWENO = np.zeros((len(hvals), 3))

        for dx in hvals:
            hs = hs + 1
            dt = dx / 2

            Nx = Nvals[hs]
            #dx = (b - a)
            Nt = int(time / dt) + 1

            x = np.linspace(a, b, Nx)  # discretization of space
            t = np.linspace(0, t, Nt)  # discretization of time

            error_1_ENO = np.zeros(len(t))
            error_2_ENO = np.zeros(len(t))
            error_inf_ENO = np.zeros(len(t))

            error_1_EENO = np.zeros(len(t))
            error_2_EENO = np.zeros(len(t))
            error_inf_EENO = np.zeros(len(t))

            error_1_WENO = np.zeros(len(t))
            error_2_WENO = np.zeros(len(t))
            error_inf_WENO = np.zeros(len(t))

            error_1_EWENO = np.zeros(len(t))
            error_2_EWENO = np.zeros(len(t))
            error_inf_EWENO = np.zeros(len(t))

            eno = ENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv)
            eno.set_initial()

            eeno = EENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv)
            eeno.set_initial()

            weno = WENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv, rr=2)
            weno.set_initial()

            eweno = EWENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv, rr=3)
            eweno.set_initial()

            t = 0
            while t < time-dt/2:
                eno.Runge_Kutta()
                eeno.Runge_Kutta()
                weno.Runge_Kutta()
                eweno.Runge_Kutta()
                t += dt

            plt.plot(x, analytic(x, time), color = 'red')
            plt.plot(x, eweno.u[3:-3], color = 'blue')
            plt.plot(x, eeno.u[5:-5], color = 'purple')
            plt.plot(x, weno.u[3:-3], color = 'green')
            plt.show()

            tempENO = abs(analytic(x, time) - eno.u[2:-2])
            tempEENO = abs(analytic(x, time) - eeno.u[5:-5])
            tempWENO = abs(analytic(x, time) - weno.u[3:-3])
            tempEWENO = abs(analytic(x, time) - eweno.u[3:-3])

            maxerror_ENO[hs, 0] = dx * sum(tempENO)
            maxerror_ENO[hs, 1] = pow(dx * sum(pow(tempENO, 2)), 1 / 2)
            maxerror_ENO[hs, 2] = max(tempENO)

            maxerror_EENO[hs, 0] = dx * sum(tempEENO)
            maxerror_EENO[hs, 1] = pow(dx * sum(pow(tempEENO, 2)), 1 / 2)
            maxerror_EENO[hs, 2] = max(tempEENO)

            maxerror_WENO[hs, 0] = dx * sum(tempWENO)
            maxerror_WENO[hs, 1] = pow(dx * sum(pow(tempWENO, 2)), 1 / 2)
            maxerror_WENO[hs, 2] = max(tempWENO)

            maxerror_EWENO[hs, 0] = dx * sum(tempEWENO)
            maxerror_EWENO[hs, 1] = pow(dx * sum(pow(tempEWENO, 2)), 1 / 2)
            maxerror_EWENO[hs, 2] = max(tempEWENO)

            print("\nMax Error when N = %d" % Nvals[hs])
            print("ENO Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_ENO[hs, i]))
            print("\t e_inf = %f" % maxerror_ENO[hs, 2])

            print("EENO Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_EENO[hs, i]))
            print("\t e_inf = %f" % maxerror_EENO[hs, 2])

            print("WENO Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_WENO[hs, i]))
            print("\t e_inf = %f" % maxerror_WENO[hs, 2])

            print("EWENO Errors:")
            for i in (0, 1):
                print("\t e_%d = %f" % (i + 1, maxerror_EWENO[hs, i]))
            print("\t e_inf = %f" % maxerror_EWENO[hs, 2])

        logH = np.log2(hvals)

        figure, axis = plt.subplots(2, 2)

        for i in (0, 1, 2):
            figplace1 = (0, 0, 1, 1)
            figplace2 = (0, 1, 0, 1)

            #slope_ENO = (-np.log2(maxerror_ENO[len(hvals) - 1, i]) + np.log2(maxerror_ENO[len(hvals) - 2, i])) / (
            #        -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            #slope_EENO = (-np.log2(maxerror_EENO[len(hvals) - 1, i]) + np.log2(maxerror_EENO[len(hvals) - 2, i])) / (
            #        -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            #slope_WENO = (-np.log2(maxerror_WENO[len(hvals) - 1, i]) + np.log2(maxerror_WENO[len(hvals) - 2, i])) / (
            #        -logH[len(hvals) - 1] + logH[len(hvals) - 2])
            #slope_EWENO = (np.log2(maxerror_EWENO[len(hvals) - 1, i]) - np.log2(maxerror_EWENO[len(hvals) - 2, i])) / (
            #        logH[len(hvals) - 1] - logH[len(hvals) - 2])
            slope_ENO = linregress(logH, np.log2(maxerror_ENO[:, i]))[0]
            slope_EENO = linregress(logH, np.log2(maxerror_EENO[:, i]))[0]
            slope_WENO = linregress(logH, np.log2(maxerror_WENO[:, i]))[0]
            slope_EWENO = linregress(logH, np.log2(maxerror_EWENO[:,i]))[0]

            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_ENO[:, i]),
                                                  label="ENO_Error - Slope %f" % slope_ENO, color="yellow")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_EENO[:, i]),
                                                  label="EENO_Error - Slope %f" % slope_EENO, color="blue")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_WENO[:, i]),
                                                  label="WENO_Error - Slope %f" % slope_WENO, color="black")
            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_EWENO[:, i]),
                                                  label="EWENO_Error - Slope %f" % slope_EWENO, color="purple")

            axis[figplace1[i], figplace2[i]].legend()
            if i != 2:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
            else:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")

        plt.show()
