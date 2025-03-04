from Solvers import solver
import matplotlib.pyplot as plt
from numpy import where
from ENO import *

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def divided_difference(f, u, x):
    if len(u) == 1:
        return f(u)
    else:
        return ( divided_difference(f, u[1:], x[1:]) - divided_difference(f, u[:-1], x[:-1])) / (x[-1] - x[0])

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
    return initial_smooth(x - a * t)

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

if __name__ == '__main__':
    analytic = analytical3

    def init(x):
        return analytic(x,0)
        #return 1/4 + 1/2*np.sin(np.pi * x)

    problem = burgers
    deriv = burgers_prime

    dx = 1/20
    dt = dx/6

    a = 0
    b = 4
    time = 2

    Nx = int((b-a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a,b,Nx)
    t = np.linspace(0, time, Nt)

    solLW = solver(dt, dx, init, (a,b), (0,time), problem, deriv, "LW")

    eno = ENO(l = a, r = b, dx=dx, dt=dt, init = init, problem = problem, deriv=deriv)
    eno.set_initial()

    eeno = EENO(l=a, r=b, dx=dx, dt=dt, init=init, problem=problem, deriv=deriv)
    eeno.set_initial()


    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    line0, = axis.plot(x, init(x), 'red', label='Analytical Solution')  # Returns a tuple of line objects, thus the comma
    line1, = axis.plot(eno.xc[2:-2], eno.u[2:-2], color = 'green', label='ENO Solution')  # Returns a tuple of line objects, thus the comma
    line2, = axis.plot(eeno.xc[4:-4], eeno.u[4:-4], color='blue', label='EENO Solution')  # Returns a tuple of line objects, thus the comma
    #lineLW, = axis.plot(x, init(x), color='purple', label='LW Solution')  # Returns a tuple of line objects, thus the comma

    plt.ylim(-.5, 2.5)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x,t)")

    text = plt.text(0, 0, "t = 0")


    for i in range(0, Nt-1):
        text.set_text("t = %f" % t[i+1])
        line0.set_ydata(analytic(x, t[i+1]))

        line1.set_ydata(eno.u[2:-2])
        eno.Runge_Kutta()

        line2.set_ydata(eeno.u[4:-4])
        eeno.Runge_Kutta()

        #lineLW.set_ydata(solLW[:, i])

        figure.canvas.draw()
        figure.canvas.flush_events()

    plt.ioff()
    plt.show()

