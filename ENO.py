import numpy as np
from numba.core.ir_utils import visit_vars
from Solvers import solver
import matplotlib.pyplot as plt
from sympy import *
from numpy import where

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def divided_difference(f, u, x):
    if len(u) == 1:
        return f(u)
    else:
        return ( divided_difference(f, u[1:], x[1:]) - divided_difference(f, u[:-1], x[:-1])) / (x[-1] - x[0])

def RK_TVD3(u, dx, L):
    u0 = u[0]
    for i in range(1, len(u[:, 0])):
        u1 = u0 + dx * L @ u0
        u2 = u0 + 1/4 * dx * L @ u0 + 1/4 * dx * L @ u1
        u[i, 1:-1] = (u0 + 1/6 * dx * L @ u0 + 1/6 *dx * L @ u1 + 2/3 * dx * L @ u2)[1:-1]
        u0 = u[i]
    return u


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
    return initial_smooth(x - a * t)



def ENO_ROE(dt, dx, init, xbounds, tbounds, f, fprime, name = "ROE"):
    # ------------ Set-Up ------------------------------
    order = 3
    Nx = int((xbounds[1] - xbounds[0]) / dx) + 1
    Nt = int((tbounds[1] - tbounds[0]) / dt) + 1
    NN = Nx + (order - 1) * 2
    sol = np.zeros((NN, Nt))
    xx = np.linspace(xbounds[0] - (order - 1) * dx, xbounds[1] + (order - 1) * dx, NN)
    sol[:, 0] = init(xx)
    ib = order - 1
    im = Nx + ib

    alpha = 1  # max(abs(fprime(xx[(order-1):(1-order)])))

    # ------------ Start iteration through time --------
    for t in range(1, Nt):
        # interface values
        u = sol[:, t - 1]
        ul = np.zeros(2 * ib + Nx + 1)
        ur = np.zeros(2 * ib + Nx + 1)
        u_m = u.copy()

        for j in range(0, 3):
            # Divided Difference, only needs to go as deep as the order of the method
            divided_diff = np.zeros((NN + 1, order+1))
            divided_diff[0:NN, 0] = u.copy()
            for k in range(1, order+1):
                divided_diff[0:NN - k, k] = (divided_diff[1:NN - (k - 1), k - 1] - divided_diff[0:NN - k, k - 1])
            ### ----------- ENO Reconstruction -----------------------
            for i in range(ib, im):
                # Make Stencil using x_i and x_(i+1)
                Roe_Speed = (f(u[i+1])- f(u[i]))/(u[i+1] - u[i])
                if Roe_Speed >= 0:
                    stencil = np.array([i-1, i])
                else:
                    stencil = np.array([i, i+1])

                z = symbols('z')
                Q = f(u[stencil[0]])*(z - xx[stencil[0]])
                for k in range(0, order):  # The number of interfaces in the stencil: k + 2
                    L, R = stencil[0], stencil[-1]

                    stencilL = np.append(L - 1, stencil)
                    stencilR = np.append(stencil, R + 1)

                    a = 1/(k+1) * divided_diff[stencilL[0], k + 1]
                    b = 1/(k+1) * divided_diff[stencilR[0], k + 1]

                    if np.abs(a) < np.abs(b):
                        prod = 1
                        c = b
                        for l in stencil:
                            prod = prod * (z - (xx[l-1] + dx))

                        stencil = stencilL.copy()
                    else:
                        prod = 1
                        c = a
                        for l in stencil:
                            prod = prod * (z - (xx[l-1] + dx))

                        stencil = stencilR.copy()

                    Q = Q + c * prod

                Q_prime = lambdify(z, Q.diff(z))

                ul[i + 1] = Q_prime(xx[i] + dx)
                ur[i] = Q_prime(xx[i] + dx)

            ul[ib] = ul[im]
            ur[im] = ur[ib]

            L = ul + ur

            ### RK Approximation
            alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
            alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
            alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]
            u[ib:im] = alpha1[j] * u_m[ib:im] + alpha2[j] * u[ib:im] - alpha3[j] * dt / dx * (
                        L[ib + 1:im + 1] - L[ib:im])

            """u1 = u0 + dx * (flux[ib+1:im+1] - flux[ib:im])
               u2 = 3/4 * u0 + 1/4 * u1 + 1/4 * dx * (flux[ib+1:im+1] - flux[ib:im])
               u[ib:im] = 1/3 * u0 + 2/3 * u2 + 2/3 * dx * (flux[ib+1:im+1] - flux[ib:im])"""

        sol[ib:im, t] = u[ib:im]

    return sol[ib:im, :]


def ENO(dt, dx, init, xbounds, tbounds, f, fprime, name = "ENO"):
    # ------------ Set-Up ------------------------------
    order = 3
    Nx = int((xbounds[1] - xbounds[0]) / dx) + 1
    Nt = int((tbounds[1] - tbounds[0]) / dt) + 1
    NN = Nx + (order - 1) * 2
    sol = np.zeros((NN, Nt))
    xx = np.linspace(xbounds[0] - (order-1)*dx, xbounds[1] + (order-1)*dx, NN)
    sol[:, 0] = init(xx)
    ib = order - 1
    im = Nx + ib

    alpha = max(abs(fprime(xx[(order-1):(1-order)])))

    # ------------ Start iteration through time --------
    for t in range(1, Nt):
        # interface values
        u = sol[:, t-1]
        ul = np.zeros(2 * ib + Nx + 1)
        ur = np.zeros(2 * ib + Nx + 1)

        # Find Crj vals for v_i+1/2 in ENO Scheme. Based on formula 2.20 in https://www3.nd.edu/~zxu2/acms60790S13/Shu-WENO-notes.pdf
        def crj_val(r):
            crj = np.zeros(order)
            for j in range(0, order):
                for m in range(j + 1, order + 1):
                    de = 1.0
                    no = 0.0
                    for l in range(order + 1):
                        if l != m:
                            de = de * (m - l)

                    for l in range(order + 1):
                        if l != m:
                            ee = 1.0
                            for q in range(order + 1):
                                if q != m and q != l:
                                    ee *= (r - q + 1)
                            no += ee

                    crj[j] += float(no) / float(de)
            return crj

        # Divided Difference, only needs to go as deep as the order of the method


        u_m = u.copy()
        for j in range(0, 3):
            divided_diff = np.zeros((NN + 1, order))
            divided_diff[0:NN, 0] = u.copy()
            for k in range(1, order):
                divided_diff[0:NN - k, k] = divided_diff[1:NN - (k - 1), k - 1] - divided_diff[0:NN - k, k - 1]
            ### ----------- ENO Reconstruction -----------------------
            for i in range(ib, im):
                # Make Stencil using x_i and x_(i+1)
                stencil = np.array([i, i+1])
                for k in range(0, order - 1): # The number of interfaces in the stencil: k + 2
                    L, R = stencil[0], stencil[-1]

                    stencilL = np.append(L-1, stencil)
                    stencilR = np.append(stencil, R+1)

                    a = divided_diff[stencilL[0], k+1]
                    b = divided_diff[stencilR[0], k+1]

                    if np.abs(a) < np.abs(b):
                        stencil = stencilL.copy()
                    else:
                        stencil = stencilR.copy()

                r = i - stencil[0] # How far a shift left is the stencil?

                cL = crj_val(r)
                cR = crj_val(r-1)

                v = u[stencil[0:-1]]

                ul[i+1] = cL @ v
                ur[i] = cR @ v

            ul[ib] = ul[im] #crj_val(2) @ u[ib-1:ib + 2]
            ur[im] = crj_val(-1) @ u[im - 1: im+2]


            flux = 1/2 * (f(ul) + f(ur) - alpha * (ur - ul))

            ### RK Approximation
            alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
            alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
            alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]
            u[ib:im] = alpha1[j] * u_m[ib:im] + alpha2[j] * u[ib:im] - alpha3[j] * dt / dx * (flux[ib + 1:im + 1] - flux[ib:im])

            """u1 = u0 + dx * (flux[ib+1:im+1] - flux[ib:im])
               u2 = 3/4 * u0 + 1/4 * u1 + 1/4 * dx * (flux[ib+1:im+1] - flux[ib:im])
               u[ib:im] = 1/3 * u0 + 2/3 * u2 + 2/3 * dx * (flux[ib+1:im+1] - flux[ib:im])"""

        sol[ib:im, t] = u[ib:im]

    return sol[ib:im, :]

if __name__ == '__main__':
    analytic = analytical2

    def init(x):
        return 1/4 + 1/2*np.sin(np.pi * x)

    problem = burgers
    deriv = burgers_prime

    dx = 1/50
    dt = 1/50

    a = -1
    b = 1
    time = 1 #2/np.pi

    Nx = int((b-a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a,b,Nx)
    t = np.linspace(0, time, Nt)

    solENO = ENO(dt, dx, init, (a,b), (0,time), problem, deriv)
    #solENO_ROE = ENO_ROE(dt, dx, init, (a, b), (0, time), problem, deriv)

    solLW = solver(dt, dx, init, (a,b), (0,time), problem, deriv, "LW")

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    #line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')  # Returns a tuple of line objects, thus the comma
    line1, = axis.plot(x, init(x), color = 'green', label='LW Solution')  # Returns a tuple of line objects, thus the comma
    line2, = axis.plot(x, init(x), color='blue', label='ENO Solution')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-.5, 1.5)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x,t)")

    text = plt.text(0, 0, "t = 0")

    for i in range(0, Nt):
        text.set_text("t = %f" % t[i])
        #line0.set_ydata(analytic(x, t[i]))
        line1.set_ydata(solLW[:, i])
        line2.set_ydata(solENO[:, i])
        figure.canvas.draw()
        figure.canvas.flush_events()

    plt.ioff()
    plt.show()

