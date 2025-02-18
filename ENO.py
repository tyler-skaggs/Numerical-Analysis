import numpy as np
import matplotlib.pyplot as plt

def burgers(x):
    return pow(x, 2) / 2

def burgers_prime(x):
    return x

def divided_difference(f, x):
    sum = 0
    for i in range(len(x)):
        prod = 0
        for j in range(len(x)):
            if j != i:
                prod *= (x[i] - x[j])
        sum += f(x[i]) / prod
    return sum

def RK_TVD3(u, dx, L):
    u0 = u[:, 0]
    n = np.shape(u)[0]
    for i in range(1, len(u[0])):
        u1 = u0 + dx * L @ u0
        u2 = u0 + 1/4 * dx * L @ u0 + 1/4 * dx * L @ u1
        u[:, i] = u0 + 1/6 * dx * L @ u0 + 1/6 *dx * L @ u1 + 2/3 * dx * L @ u2
        u0 = u[:, i]
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


if __name__ == '__main__':
    analytic = analytical2

    def init(x):
        return analytic(x, 0)

    problem = burgers
    deriv = burgers_prime

    dx = 1/10
    dt = 1/10

    a = 0
    b = 1
    time = 10

    Nx = int((b-a) / dx)
    Nt = int(time / dt)

    x = np.linspace(a,b,Nx)
    t = np.linspace(0, time, Nt)

    u = np.zeros((Nt, Nx))

    u[0] = np.ones(Nx)

    print(u)
    L = np.zeros((Nt,Nt))
    for i in range(Nt):
        L[i,i] = -2
        if i == 0:
            L[i, i+1] = 1

        elif i == Nt-1:
            L[i, i-1] = 1

        else:
            L[i, i+1] = 1
            L[i, i-1] = 1

    L = L * 1 / pow(dx,2)

    u = RK_TVD3(u, dt, L)

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    line1, = axis.plot(x, u[0], color = 'blue', label='Numerical Solution')  # Returns a tuple of line objects, thus the comma
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x,t)")

    text = plt.text(0, 0, "t = 0")
    print(u)
    for i in range(1, Nt):
        text.set_text("t = %f" % t[i])
        #line0.set_ydata(analytic(x, t[i]))
        line1.set_ydata(u[0])
        figure.canvas.draw()
        figure.canvas.flush_events()


