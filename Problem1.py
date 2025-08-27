import matplotlib.pyplot as plt
import numpy as np
from numpy import where
from scipy.stats import linregress


def init(x, lam):
    return 1/(1 + np.exp(lam * x))


if __name__ == '__main__':

    dx = 1 / 20
    dt = dx / 2

    a = -10
    b = 20
    time = 10

    k = [1, 1.2, 2.5, 3]
    lam = [0.49, 0.45, 0.34, 0.8]

    Nx = int((b - a) / dx) + 1
    Nt = int(time / dt) + 1

    x = np.linspace(a, b, Nx)
    t = np.linspace(0, time, Nt)

    plt.ion()
    figure = plt.figure()
    axis = figure.add_subplot(111)

    U = [init(x, lam[0]), init(x, lam[1]), init(x, lam[2]), init(x, lam[3])]
    L = [U[0][0], U[1][0], U[2][0], U[3][0]]
    print(L)
    R = [U[0][-1], U[1][-1], U[2][-1], U[3][-1]]

    lineSOL0, = axis.plot(x, U[0], color='red', label=r'$k = 1, \ \lambda = 0.49$')
    lineSOL1, = axis.plot(x, U[1], color='blue', label=r'$k = 1.2, \ \lambda = 0.45$')
    lineSOL2, = axis.plot(x, U[2], color='green', label=r'$k = 2.5, \ \lambda = 0.34$')
    lineSOL3, = axis.plot(x, U[3], color='orange', label=r'$k = 3, \ \lambda = 0.8$')

    plt.ylim(-0.5, 2.5)
    #plt.xlim(-10, 50)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x,t)")
    plt.title(r'Numerical Solution to $u_t - ku_{xxt} = u_{xx} +u(1-u), \quad \Delta x = \frac{1}{20}, \ \frac{\Delta t}{\Delta x} = \frac{1}{2}$')

    text = plt.text(14, 1.5, "t = 0")

    Dxx = -2 * np.eye(Nx)
    for i in range(0, Nx - 1):
        Dxx[i, i + 1] = Dxx[i + 1, i] = 1

    # Forward Difference for Left Endpoint
    Dxx[0,0] = 1
    Dxx[0,1] = -2
    Dxx[0,2] = 1

    # Backward Difference for Right Endpoint
    Dxx[-1, -1] = 1
    Dxx[-1, -2] = -2
    Dxx[-1, -3] = 1

    Dxx *= np.power(dx,-2)

    M = [np.eye(Nx) - k[0] * Dxx, np.eye(Nx) - k[1] * Dxx, np.eye(Nx) - k[2] * Dxx, np.eye(Nx) - k[3] * Dxx]
    Minv = [np.linalg.inv(M[0]), np.linalg.inv(M[1]), np.linalg.inv(M[2]), np.linalg.inv(M[3])]

    t = 0
    i=0
    while t < time-dt/2:
        for j in range(4):
            V = M[j] @ U[j]
            V += dt * (-1 / k[j] * V + (1 / k[j] + 1) * U[j] - np.power(U[j], 2))  # update V

            U[j] = Minv[j] @ V # Update U
            U[j][-1] = R[j]


        t += dt
        text.set_text("t = %f" % t)

        lineSOL0.set_ydata(U[0])
        lineSOL1.set_ydata(U[1])
        lineSOL2.set_ydata(U[2])
        lineSOL3.set_ydata(U[3])

        figure.canvas.draw()
        figure.canvas.flush_events()
        i+=1


    plt.ioff()
    plt.show()