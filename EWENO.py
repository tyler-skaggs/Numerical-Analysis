import matplotlib.pyplot as plt
import numpy as np
from numpy import where
from scipy.stats import linregress

from temp import analytical1, analytical2, analytical3


def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def linear_deriv(x):
    if type(x) == type(1.0):
        return 1
    else:
        return np.ones(len(x))

def EWENO(x, u, dx, dt, problem, deriv, r):
    unew = u.copy()

    a = np.array([[[-1 / 2, 3 / 2, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]],
                  [[1 / 3, -7 / 6, 11 / 6], [-1 / 6, 5 / 6, 1 / 3], [1 / 3, 5 / 6, -1 / 6]]])
    e = pow(10, -6)
    f_plushalf = np.zeros(len(x))


    def q(k, r, g):
        sum = 0
        for l in range(len(g)):
            sum += a[r - 2, k, l] * g[l]
        return sum

    def L(u):
        #Set Ghost Cells
        #Left
        for k in range(1, r):
            u[r - k] = u[len(x)-r - k-1]

        # right boundary
        for k in range(r):
            u[len(x) - r + k] = u[r + k+1]

        fl = u.copy()
        for j in range(len(u)-1):
            ran = deriv(np.linspace(u[j], u[j+1], 100))
            alpha = max(abs(ran))
            if min(ran) > 0:
                fl[j] = problem(u[j])
            elif max(ran) < 0:
                fl[j] = problem(u[j+1])
            else:
                fl[j] = 1/2 * (problem(u[j]) + problem(u[j+1]) - alpha * (u[j+1] - u[j]))

        for j in range(r-1, len(x) - r + 1):
            IS0 = ( 13/12 * np.power(fl[j-2] - 2 * fl[j-1] + fl[j], 2) +
                   1/4 * np.power(fl[j-2] - 4*fl[j-1] + 3*fl[j], 2) )

            IS1 = ( 13/12 * np.power(fl[j-1] - 2*fl[j] + fl[j+1], 2) +
                   1/4 * np.power(fl[j-1] - fl[j+1], 2) )

            IS2 = ( 13/12 * np.power(fl[j] - 2 * fl[j+1] + fl[j+2], 2) +
                    1/4 * np.power(3*fl[j] - 4*fl[j+1] +fl[j+2], 2) )

            alphas = np.array([(1/10) / np.power(e + IS0, 3),
                               (6/10) / np.power(e + IS1, 3),
                               (3/10) / np.power(e + IS2, 3)])


            omega = alphas/np.sum(alphas)

            sum = 0
            for k in range(r):
                sum += omega[k] * q(k, r, fl[(j + k - r + 1):(j+k+1)])

            f_plushalf[j] = sum

        L = -1/dx * (f_plushalf[r:-r] - f_plushalf[r-1:-r-1])

        return L

    """alpha0 = [1, 3/4, 3/8, 1/4, 89537/2880000, 4/9] #[1, 1, 1, -1/3, 0, 0]
    alpha1 = [0, 1/4, 1/8, 1/8, 407023/2880000, 1/15] #[0, 0, 0, 1/3, 0, 0]
    alpha2 = [0, 0, 1/2, 1/8, 1511/12000, 0] #[0, 0, 0, 2/3, 0, 0]
    alpha3 = [0, 0, 0, 1/2, 87/200, 8/45] #[0, 0, 0, 1/3, 0, 0]
    alpha4 = [0, 0, 0, 0, 4/15, 0] #[0, 0, 0, 0, 0, 0]
    alpha5 = [0, 0, 0, 0, 0, 14/45] #[0, 0, 0, 0, 0, 0]
    alphaL0 = [1/2, 0, -1/8, -5/64, 2276219/40320000, 0] #[1/2, 0, 0, 0, 0, 0]
    alphaL1 = [0, 1/8, -1/16, -13/64, 407023/672000, -8/45] #[0, 1/2, 0, 0, 0, 0]
    alphaL2 = [0, 0, 1/2, 1/8, 1511/2800, 0] #[0, 0, 1, 0, 0, 0]
    alphaL3 = [0, 0, 0, 9/16, -261/140, 2/3] #[0, 0, 0, 1/6, 0, 0]
    alphaL4 = [0, 0, 0, 0, 8/7, 0] #[0, 0, 0, 0, 0, 0]
    alphaL5 = [0, 0, 0, 0, 0, 7/90] #[0, 0, 0, 0, 0, 0]"""

    alpha0 = [1, 1, 1, -1/3, 0, 0]
    alpha1 = [0, 0, 0, 1/3, 0, 0]
    alpha2 = [0, 0, 0, 2/3, 0, 0]
    alpha3 = [0, 0, 0, 1/3, 0, 0]
    alpha4 = [0, 0, 0, 0, 1, 0]
    alpha5 = [0, 0, 0, 0, 0, 1]
    alphaL0 = [1/2, 0, 0, 0, 0, 0]
    alphaL1 = [0, 1/2, 0, 0, 0, 0]
    alphaL2 = [0, 0, 1, 0, 0, 0]
    alphaL3 = [0, 0, 0, 1/6, 0, 0]
    alphaL4 = [0, 0, 0, 0, 0, 0]
    alphaL5 = [0, 0, 0, 0, 0, 0]


    uk = [u.copy(), u.copy(), u.copy(), u.copy(), u.copy(), u.copy(), u.copy()]
    Lk = [u.copy(), u.copy(), u.copy(), u.copy(), u.copy(), u.copy()]
    for k in range(6):
        Lk[k][r:-r] = L(unew)
        uk[k + 1][r:-r] = \
            (alpha0[k] * uk[0][r:-r] +
             alpha1[k] * uk[1][r:-r] +
             alpha2[k] * uk[2][r:-r] +
             alpha3[k] * uk[3][r:-r] +
             alpha4[k] * uk[4][r:-r] +
             alpha5[k] * uk[5][r:-r] +
             alphaL0[k] * dt * Lk[0][r:-r] +
             alphaL1[k] * dt * Lk[1][r:-r] +
             alphaL2[k] * dt * Lk[2][r:-r] +
             alphaL3[k] * dt * Lk[3][r:-r] +
             alphaL4[k] * dt * Lk[4][r:-r] +
             alphaL5[k] * dt * Lk[5][r:-r])

        unew = uk[k + 1].copy()

    return unew

if __name__ == '__main__':

    def init(x):
        return np.sin(np.pi * x)**4
        #return analytic(x,0)

    def analytic(x,t):
        return init(x - t)


    problem = burgers_prime
    deriv = linear_deriv

    dx = 1/20
    dt = dx/2 #np.power(dx, 5/4)

    a = -1
    b = 1
    time = 1
    r = 3

    Nx = int((b-a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a, b, Nx)
    xc = np.linspace(a - r * dx, b + r * dx, 2*r + Nx)
    t = np.linspace(0, time, Nt)

    plot = 0

    if plot == 1:

        ewenoSOL = init(xc)

        plt.ion()
        figure = plt.figure()
        axis = figure.add_subplot(111)

        line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')
        lineEWENO, = axis.plot(x, ewenoSOL[r:-r], color='purple', label='E-WENO Solution')

        plt.ylim(-0.5, 2.5)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x,t)")

        text = plt.text(0, 0, "t = 0")

        t = 0
        i = 0
        while t < time-dt/2:
            ewenoSOL = EWENO(xc, ewenoSOL, dx, dt, problem, deriv, r)
            t += dt
            text.set_text("t = %f" % t)

            line0.set_ydata(analytic(x, t))
            lineEWENO.set_ydata(ewenoSOL[r:-r])

            figure.canvas.draw()
            figure.canvas.flush_events()

            i+=1
        plt.ioff()
        plt.show()

        error1 = dx * sum(abs(ewenoSOL[r:-r] - analytic(x, t)))
        errorinf = max(abs(ewenoSOL[r:-r] - analytic(x, t)))
        print("N = ", len(x), "\n\tL1 Error: ", error1, '\n\tLinf Error: ', errorinf)

    else:
        Nvals = np.array([80, 160, 320])
        hvals = (b - a) / (Nvals - 1)
        hs = -1
        maxerror_EWENO = np.zeros((len(hvals), 3))

        for dx in hvals:
            hs = hs + 1
            dt = dx/2

            Nx = Nvals[hs]
            Nt = int(time / dt) + 1

            x = np.linspace(a, b, Nx)  # discretization of space
            xc = np.linspace(a - r * dx, b + r * dx, 2*r + Nx)
            t = np.linspace(0, t, Nt)  # discretization of time

            error_1_EWENO = np.zeros(len(t))
            error_2_EWENO = np.zeros(len(t))
            error_inf_EWENO = np.zeros(len(t))
            eweno = init(xc)

            t = 0
            while t < time:
                eweno = EWENO(xc, eweno, dx, dt, problem, deriv, r)
                t += dt

            plt.plot(x, analytic(x, t), color='red', label="Analytic")
            plt.plot(x, eweno[3:-3], color='purple', label='E-WENO')
            plt.legend()
            plt.show()

            tempEWENO = abs(analytic(x, t) - eweno[r:-r])

            maxerror_EWENO[hs, 0] = dx * sum(tempEWENO)
            maxerror_EWENO[hs, 1] = pow(dx * sum(pow(tempEWENO, 2)), 1 / 2)
            maxerror_EWENO[hs, 2] = max(tempEWENO)

        logH = np.log2(hvals)
        figure, axis = plt.subplots(2, 2)

        for i in (0, 1, 2):
            figplace1 = (0, 0, 1, 1)
            figplace2 = (0, 1, 0, 1)

            slope_EWENO = linregress(logH, np.log2(maxerror_EWENO[:, i]))[0]

            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_EWENO[:, i]),
                                                  label="EWENO_Error - Slope %f" % slope_EWENO, color="purple")

            axis[figplace1[i], figplace2[i]].legend()
            if i != 2:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
            else:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")
        plt.show()

        print('N    ', 'L_inf                     ', 'Ord                         ', 'L1                        ', 'Ord')
        print(f"{Nvals[0]:3d}", ' ', f"{maxerror_EWENO[0, 2]:3.22f}", '                            | ', f"{maxerror_EWENO[0, 0]:3.22f}")
        for i in range(1, len(Nvals)):
            ord_inf = (-np.log2(maxerror_EWENO[i, 2]) + np.log2(maxerror_EWENO[i-1, 2])) / (-logH[i] + logH[i-1])
            ord_1 = (-np.log2(maxerror_EWENO[i, 0]) + np.log2(maxerror_EWENO[i-1, 0])) / (-logH[i] + logH[i-1])
            print(f"{Nvals[i]:3d}", ' ', f"{maxerror_EWENO[i, 2]:3.22f}", ' ', f"{ord_inf:3.22f}", ' | ', f"{maxerror_EWENO[i, 0]:3.22f}", ' ', f"{ord_1:3.22f}")

