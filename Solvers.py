import numpy as np
import scipy.optimize as opt

def solver(k, h, init, xbound, tbound, f, fprime, name):
    Nx = int((xbound[1] - xbound[0]) / h) + 1
    Nt = int((tbound[1] - tbound[0]) / k) + 1

    x = np.linspace(xbound[0], xbound[1], Nx)
    y = init(x)
    temp = init(x)
    sol = np.zeros((Nx, Nt))
    for t in range(0, Nt):
        sol[:, t] = y
        if name == "R":
            temp[1:-1] = 1 / 2 * ( y[1:-1] + y[2:] ) - k / (2 * h) * (f(y[2:]) - f(y[1:-1]))
            y[1:-1] = y[1:-1] - k/h * ( f(temp[1:-1]) - f(temp[:-2]) )

        elif name == "LF":
            def Flux(v,w):
                F = np.zeros(len(v))
                F = 1/2 * (f(v) + f(w)) - h / (2 * k) * (w - v)
                return F
            y[1:-1] = y[1:-1] - k / h * (Flux(y[1:-1], y[2:]) - Flux(y[:-2], y[1:-1]))

        elif name == "U":
            def Flux(v, w):
                F = np.zeros(len(v))
                for i in range(0, len(v)):
                    tempv = 0
                    if v[i] == w[i]:
                        tempv = k / h * fprime(v[i])
                    else:
                        tempv = k / h * (f(w[i]) - f(v[i])) / (w[i] - v[i])

                    if tempv > 0:
                        F[i] = f(v[i])
                    else:
                        F[i] = f(w[i])
                return F

            y[1:-1] = y[1:-1] - k / h * (Flux(y[1:-1], y[2:]) - Flux(y[:-2], y[1:-1]))

        elif name == "G":
            def Flux(v, w):
                F = np.zeros(len(v))
                for i in range(0, len(v)):
                    if v[i] == w[i]:
                        F[i] = f(v[i])
                    elif v[i] < w[i]:
                        F[i] = f(opt.fminbound(lambda x: f(x), v[i], w[i]))
                    else:
                        F[i] = f(opt.fminbound(lambda x: -f(x), w[i], v[i]))
                return F

            y[1:-1] = y[1:-1] - k / h * (Flux(y[1:-1], y[2:]) - Flux(y[:-2], y[1:-1]))

        elif name == "LW":
            def Flux(v,w):
                F = np.zeros(len(v))
                for i in range(0, len(v)):
                    tempv = 0
                    if v[i] == w[i]:
                        tempv = k / h * fprime(v[i])
                    else:
                        tempv = k / h * (f(w[i]) - f(v[i])) / (w[i] - v[i])

                    F[i] = 1/2 * ((f(w[i]) + f(v[i])) - tempv * (f(w[i]) - f(v[i])))
                return F

            y[1:-1] = y[1:-1] - k / h * (Flux(y[1:-1], y[2:]) - Flux(y[:-2], y[1:-1]))

        elif name == "M":
            temp[1:-1] = y[1:-1] - k / h * (f(y[2:]) - f(y[1:-1]))
            y[1:-1] = 1 / 2 * (y[1:-1] + temp[1:-1]) - k / (2 * h) * (f(temp[1:-1]) - f(temp[:-2]))

        elif name == 'CD':
            def MM(x, y):
                vals = x.copy()
                for i in range(len(x)):
                    vals[i] = 1 / 2 * (np.sign(x[i]) + np.sign(y[i])) * min(np.abs(x[i]), np.abs(y[i]))
                return vals

            vp = MM(y[2:] - y[1:-1], y[1:-1] - y[:-2])
            fp = MM(f(y[2:]) - f(y[1:-1]), f(y[1:-1]) - f(y[:-2]))
            g = f(y[1:-1] - 1/2 * k/h * fp) + 1/8 * h/k * vp
            g = np.append(g, f(y[-1] - 1/2 * k/h * (f(y[-1]) - f(y[-2]))) + 1/8 * h/k * (y[-1] - y[-2]))
            g = np.append(f(y[1] - 1 / 2 * k / h * (f(y[1]) - f(y[0]))) + 1 / 8 * h / k * (y[1] - y[0]), g)

            tempL = 1/2 * (y[:-2] + y[1:-1]) - k/h * (g[1:-1] - g[:-2])
            tempR = 1/2 * (y[1:-1] + y[2:]) - k/h * (g[2:] - g[1:-1])

            y[1:-1] = (tempL + tempR) / 2

        else:
            print("Please enter valid method name")
            return 0

    return sol