import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy.stats import linregress


def burgers(z):
    return pow(z, 2) / 2

def burgers_prime(z):
    return z

def linear_deriv(z):
    if type(z) == type(1.0):
        return 1
    else:
        return np.ones(len(z))

def v(x, j, l, xarr, dx): #Legendre Polynomials
    if l == 0:
        return 1
    elif l == 1:
        return x - xarr[j]
    else:
        return np.power(x - xarr[j], 2) - 1/12 * np.power(dx, 2)

def DG(x, u0, u1, u2, dx, dt, problem, deriv, order=3):
    def hRF(a, b):  # Roe Flux with entropy fix
        low = min(a, b)
        high = max(a, b)

        if sum(deriv(np.linspace(low, high, 100) == 1)) == 100:
            maxder = 1
            minder = 1
        else:
            maxder = np.round(deriv(optimize.minimize_scalar(lambda z: -deriv(z), bounds=[low, high], method='bounded').x), 5)
            minder = np.round(deriv(optimize.minimize_scalar(lambda z: deriv(z), bounds=[low, high], method='bounded').x), 5)

        if minder >= 0:
            return problem(a)
        elif maxder <= 0:
            return problem(b)
        else:
            beta = np.abs(deriv(optimize.minimize_scalar(lambda z: -np.abs(deriv(z)), bounds=[low, high], method='bounded').x))
            return 1 / 2 * (problem(a) + problem(b) - beta * (b - a))

    def mm(arr, j): # modified minmod function
        M2 = np.pi**2
        M = 2/9 * (3 + 10 * M2) * M2 * (np.power(dx, 2) / (np.power(dx,2) + np.abs(u0[j+1] - u0[j]) + np.abs(u0[j] - u0[j-1])) )
        def m(arr): # minmod function
            if sum(arr < 0) == 0:
                return min(abs(arr))
            elif sum(arr > 0) == 0:
                return -min(abs(arr))
            else:
                return 0

        if abs(arr[0]) <= M * np.power(dx,2):
            return arr[0]
        else:
            return m(arr)

    def L(u0, u1, u2):
        left = 0
        right = 0
        for i in range(2):  # left bound
            u0[i] = left #u0[-3 + i]
            u1[i] = left #u1[-3 + i]
            u2[i] = left #u2[-3 + i]

        for i in range(2):  # right bound
            u0[-2 + i] = right #u0[1 + i]
            u1[-2 + i] = right #u1[1 + i]
            u2[-2 + i] = right #u2[1 + i]

        n = len(u0)
        L0, L1, L2 = np.zeros(n), np.zeros(n), np.zeros(n)

        if order == 3:
            utilde = 6*u1 + 30*u2
            uttilde = 6*u1 - 30*u2
        else:
            utilde = 6*u1
            uttilde = 6*u1

        utilde_mod = utilde
        uttilde_mod = uttilde

        uminus_mod = np.zeros(n)
        uplus_mod = np.zeros(n)

        for j in range(1, n-1):
            utilde_mod[j] = mm(np.array([utilde[j], u0[j+1] - u0[j], u0[j] - u0[j-1]]), j)
            uttilde_mod[j] = mm(np.array([uttilde[j], u0[j+1] - u0[j], u0[j] - u0[j-1]]), j)

            uminus_mod[j] = u0[j] + utilde_mod[j]
            uplus_mod[j-1] = u0[j] - uttilde_mod[j]

        u1mod = 1/12*(utilde_mod + uttilde_mod)
        u2mod = 1/60*(utilde_mod - uttilde_mod)

        hPlusHalf = np.zeros(n)

        for j in range(1, n-1):
            hPlusHalf[j] = hRF(uminus_mod[j], uplus_mod[j])

        hPlusHalf[0] = uplus_mod[0]
        hPlusHalf[-1] = uminus_mod[-1]


        def U(val):
            a = [1, 12 / dx, 180 / np.power(dx, 2)]
            j = sum(x <= val)
            if not(x[j] + dx / 2 >= val >= x[j] - dx / 2):
                j = j - 1

            a0 = a[0] * u0[j]
            a1 = a[1] * u1mod[j] * (val - x[j])
            if order == 3:
                a2 = a[2] * u2mod[j] * (np.power(val - x[j], 2) - 1/12 * np.power(dx, 2) )
            else:
                a2 = 0
            return  a0 + a1 + a2

        for j in range(1, n-1):
            # Function Evals for GL approximation
            vals = np.array([(14 - np.sqrt(7)) / 30 * problem(U(dx / 2 * np.sqrt(1 / 3 + 2 * np.sqrt(7)/21) + x[j])),
                             (14 - np.sqrt(7)) / 30 * problem(U(-dx / 2 * np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21) + x[j])),
                             (14 + np.sqrt(7)) / 30 * problem(U(dx / 2 * np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21) + x[j])),
                             (14 + np.sqrt(7)) / 30 * problem(U(-dx / 2 * np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21) + x[j]))])

            # d/dt u_j^(0)
            L0[j] = -1 / dx * (hPlusHalf[j] - hPlusHalf[j-1])

            #d/dt u_j^(1)
            GLint1 = 1/2 * (1/15 * (hPlusHalf[j] + hPlusHalf[j-1]) +
                             sum(vals)
                             )#Gauss-Lobotto 6 point evaluation


            L1[j] = -1/(2 * dx) * (hPlusHalf[j] + hPlusHalf[j-1]) + 1/dx * GLint1

            #d/dt u_j^(2)
            GLint2 = (1/15 * hPlusHalf[j] * dx/2 +
                      1/15 * hPlusHalf[j-1] * -dx/2 +
                      vals[0] * (dx / 2 * np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21)) +
                      vals[1] * (-dx / 2 * np.sqrt(1 / 3 + 2 * np.sqrt(7) / 21)) +
                      vals[2] * (dx / 2 * np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21)) +
                      vals[3] * (-dx / 2 * np.sqrt(1 / 3 - 2 * np.sqrt(7) / 21))
                      )


            L2[j] = -1/(6 * dx) *  (hPlusHalf[j] - hPlusHalf[j-1]) + 1/(np.power(dx, 2)) * GLint2

        return L0, L1, L2

    uk = np.array([[u0.copy(), u0.copy(), u0.copy(), u0.copy()],
                   [u1.copy(), u1.copy(), u1.copy(), u1.copy()],
                   [u2.copy(), u2.copy(), u2.copy(), u2.copy()]])

    Lk = L(uk[0, 0], uk[1, 0], uk[2, 0])
    for l in range(3):
        uk[l, 1][1:-1] = uk[l,0][1:-1] + dt * Lk[l][1:-1]

    Lk = L(uk[0, 1], uk[1, 1], uk[2, 1])
    for l in range(3):
        uk[l, 2][1:-1] = 3/4 * uk[l,0][1:-1] + 1/4*uk[l, 1][1:-1] + dt/4 * Lk[l][1:-1]

    Lk = L(uk[0, 2], uk[1, 2], uk[2, 2])
    for l in range(3):
        uk[l, 3][1:-1] = 1/3 * uk[l,0][1:-1] + 2/3*uk[l,2][1:-1] + 2/3*dt*Lk[l][1:-1]


    for i in range(2):
        for l in range(3):
            uk[l, -1][i] = 0 #uk[l, -1][-3 + i]

    return uk[0, -1], uk[1, -1], uk[2, -1]

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

if __name__ == '__main__':
    def init(x):
        return np.sin(-np.pi * x)
        return analytical2(x, 0)

    def analytic(x,t):
        #return init(x - t)
        return analytical3(x,t)

    order = 2
    plot = 1

    problem = burgers
    deriv = burgers_prime

    dx = 1/10000
    dt = dx/3 #np.power(dx, 5/4)

    a = -1
    b = 1
    time = 2/np.pi

    Nx = int((b-a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a-dx, b+dx, Nx+2)
    t = np.linspace(0, time, Nt)


    if plot == 1:
        DGSOL = x.copy()

        plt.ion()
        figure = plt.figure()
        axis = figure.add_subplot(111)

        #line0, = axis.plot(x[1:-1], init(x[1:-1]), 'red', label='Analytical Solution')
        lineDG, = axis.plot(x[1:-1], init(x[1:-1]), color='purple', label='DGRK Solution')
        #lineSOL, = axis.plot(x[1:-1], init(x[1:-1]), color='blue', label='DGRK Func Solution')


        plt.ylim(-1.2, 1.2)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x,t)")

        text = plt.text(0, 0, "t = 0")

        t = 0
        i = 0

        u0 = init(x)
        u1 = init(x)
        u2 = init(x)


        for j in range(0, len(u0)):
            lowbound = x[j] - dx/2
            highbound = x[j] + dx/2

            u0[j] = 1/dx * integrate.fixed_quad(init, lowbound, highbound, n = 6)[0]

            u1[j] = 1/np.power(dx,2) * integrate.fixed_quad(lambda z: init(z)*(z - x[j]), lowbound, highbound, n = 6)[0]

            u2[j] = 1/np.power(dx,3) * integrate.fixed_quad(lambda z: init(z)*(np.power(z-x[j], 2) - 1/12 * np.power(dx, 2)), lowbound, highbound, n = 6)[0]


        def Uu(val, u0_, u1_, u2_):
            a = [1, 12 / dx, 180 / np.power(dx, 2)]
            j = sum(x <= val)
            if not (x[j] + dx / 2 >= val >= x[j] - dx / 2):
                j = j - 1

            a0 = a[0] * u0_[j]
            a1 = a[1] * u1_[j] * (val-x[j])
            if order == 3:
                a2 = a[2] * u2_[j] * (np.power(val - x[j], 2) - 1 / 12 * np.power(dx, 2) )
            else:
                a2 = 0
            return a0 + a1 + a2

        while t < time - dt/2:
            u0, u1, u2 = DG(x, u0, u1, u2, dx, dt, problem, deriv, order)

            for j in range(0, len(x) - 1):
                DGSOL[j] = Uu(x[j], u0, u1, u2)

            t += dt
            text.set_text("t = %f" % t)

            #line0.set_ydata(analytic(x[1:-1], t ))
            lineDG.set_ydata(u0[1:-1])
            #lineSOL.set_ydata(DGSOL[1:-1])

            figure.canvas.draw()
            figure.canvas.flush_events()

            i+=1
        plt.ioff()
        plt.show()

        """error1 = dx * sum(abs(DGSOL - analytic(x, t)))
        errorinf = max(abs(DGSOL - analytic(x, t)))
        print("N = ", len(x), "\n\tL1 Error: ", error1, '\n\tLinf Error: ', errorinf)"""

    else:
        Nvals = np.array([5, 10, 20, 40, 80, 160])
        hvals = (b - a) / (Nvals - 1)
        hs = -1
        maxerror_DG = np.zeros((len(hvals), 3))

        for dx in hvals:
            hs = hs + 1
            dt = dx / 4

            Nx = Nvals[hs]
            Nt = int(time / dt) + 1

            x = np.linspace(a - dx, b + dx, Nx + 2)
            t = np.linspace(0, time, Nt)

            error_1_DG = np.zeros(len(t))
            error_2_DG = np.zeros(len(t))
            error_inf_DG = np.zeros(len(t))

            t = 0
            i = 0

            u0 = init(x)
            u1 = init(x)
            u2 = init(x)

            for j in range(0, len(u0)):
                lowbound = x[j] - dx / 2
                highbound = x[j] + dx / 2

                u0[j] = 1 / dx * integrate.fixed_quad(init, lowbound, highbound, n=6)[0]

                u1[j] = 1 / np.power(dx, 2) * \
                        integrate.fixed_quad(lambda z: init(z) * (z - x[j]), lowbound, highbound, n=6)[0]

                u2[j] = 1 / np.power(dx, 3) * \
                        integrate.fixed_quad(lambda z: init(z) * (np.power(z - x[j], 2) - 1 / 12 * np.power(dx, 2)),
                                             lowbound, highbound, n=6)[0]

            DGSOL = x.copy()
            def Uu(val, u0_, u1_, u2_):
                a = [1, 12 / dx, 180 / np.power(dx, 2)]
                j = sum(x <= val)
                if not (x[j] + dx / 2 >= val >= x[j] - dx / 2):
                    j = j - 1

                a0 = a[0] * u0_[j]
                a1 = a[1] * u1_[j] * (val - x[j])
                if order == 3:
                    a2 = a[2] * u2_[j] * (np.power(val - x[j], 2) - 1 / 12 * np.power(dx, 2))
                else:
                    a2 = 0

                return a0 + a1 + a2

            while t < time-dt/2:
                u0, u1, u2 = DG(x, u0, u1, u2, dx, dt, problem, deriv, order)
                t += dt

            for j in range(0, len(x) - 1):
                DGSOL[j] = Uu(x[j], u0, u1, u2)

            plt.plot(x[1:-1], analytic(x[1:-1], t), color='red', label="Analytic")
            plt.plot(x[1:-1], DGSOL[1:-1], color='purple', label='DG Func Solution')
            plt.plot(x[1:-1], u0[1:-1], color='purple', label='DG Solution')
            plt.legend()
            plt.show()

            tempError = abs(analytic(x[1:-1], t) - u0[1:-1])

            maxerror_DG[hs, 0] = dx * sum(tempError)
            maxerror_DG[hs, 1] = pow(dx * sum(pow(tempError, 2)), 1 / 2)
            maxerror_DG[hs, 2] = max(tempError)

        logH = np.log2(hvals)
        figure, axis = plt.subplots(2, 2)

        for i in (0, 1, 2):
            figplace1 = (0, 0, 1, 1)
            figplace2 = (0, 1, 0, 1)

            slope_EWENO = linregress(logH, np.log2(maxerror_DG[:, i]))[0]

            axis[figplace1[i], figplace2[i]].plot(-logH, -np.log2(maxerror_DG[:, i]),
                                                  label="EWENO_Error - Slope %f" % slope_EWENO, color="purple")

            axis[figplace1[i], figplace2[i]].legend()
            if i != 2:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_%d$" % (i + 1))
            else:
                axis[figplace1[i], figplace2[i]].set_title(r"$e_\infty$")
        plt.show()

        print('N    ', 'L_inf                     ', 'Ord                         ', 'L1                        ', 'Ord')
        print(f"{Nvals[0]:3d}", ' ', f"{maxerror_DG[0, 2]:3.22f}", '                            | ', f"{maxerror_DG[0, 0]:3.22f}")
        for i in range(1, len(Nvals)):
            ord_inf = (-np.log2(maxerror_DG[i, 2]) + np.log2(maxerror_DG[i - 1, 2])) / (-logH[i] + logH[i - 1])
            ord_1 = (-np.log2(maxerror_DG[i, 0]) + np.log2(maxerror_DG[i - 1, 0])) / (-logH[i] + logH[i - 1])
            print(f"{Nvals[i]:3d}", ' ', f"{maxerror_DG[i, 2]:3.22f}", ' ', f"{ord_inf:3.22f}", ' | ', f"{maxerror_DG[i, 0]:3.22f}", ' ', f"{ord_1:3.22f}")