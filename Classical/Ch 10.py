import numpy as np
import matplotlib.pyplot as plt
from numpy import where


def backward_euler(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 5, int(5/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[1:-1] = y[1:-1] - (k / (2 * h)) * A * (y[2:] - y[:-2])


def one_sided_left(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[1:-1] = y[1:-1] - (k / h) * A * (y[1:-1] - y[:-2])

def one_sided_right(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[1:-1] = y[1:-1] - (k / h) * A * (y[2:] - y[1:-1])


def lax_friedrichs(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[1:-1] = 1 / 2 * (y[:-2] + y[2:]) - k / (2*h) * A * (y[2:] - y[:-2])


def leapfrog(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    yt = y
    prev = y
    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        yt = y
        if(step == 0):
            y[1:-1] = 1 / 2 * (y[:-2] + y[2:]) - k / (2*h) * A * (y[2:] - y[:-2])
        else:
            y[1:-1] = prev[1:-1] - k / (2 * h) * A * (y[2:] - y[:-2])

        prev = yt


def lax_wendroff(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[1:-1] = y[1:-1] - (k / (2 * h)) * A * (y[2:] - y[:-2]) + (k * k / (2 * h * h)) * A * A * (y[2:] - 2 * y[1:-1] + y[:-2])


def beam_warming(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[2:] = y[2:] - k / (2 * h) * A * (3 * y[2:] - 4 * y[1:-1] + y[:-2]) + k * k / (2 * h * h) * A * A * (y[2:] - 2 * y[1:-1] + y[:-2])


def beam_warming_right(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[:-2] = y[:-2] - k / (2 * h) * A * (3 * y[:-2] - 4 * y[1:-1] + y[2:]) + k * k / (2 * h * h) * A * A * (y[:-2] - 2 * y[1:-1] + y[2:])


def extra(A, init, h, k, xbound):
    if(h < 2*k):
        k = h / 2

    N = int((xbound[1] - xbound[0]) / h) + 1
    x = np.linspace(xbound[0], xbound[1], N)
    y = init(x)

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    line1, = ax.plot(x, y, 'r-')  # Returns a tuple of line objects, thus the comma
    plt.ylim(-0.5, 1.5)

    for step in np.linspace(0, 1, int(1/k)):
        line1.set_ydata(y)
        fig.canvas.draw()
        fig.canvas.flush_events()

        y[2:-2] = y[2:-2] - (k / (2 * h)) * A * (y[3:-1] - y[1:-3]) + (k * k / (2 * h * h)) * A * A * (y[3:-1] - 2 * y[2:-2] + y[1:-3]) - k * k * k /(6 * h * h * h) * A * A * A * (y[4:] - 2 * y[3:-1] + 2 * y[1:-3] - y[:-4])


def initial_condition1(x):
    y = np.zeros(np.size(x))
    for i in range(0, np.size(x)):
        if(x[i] <= 0):
            y[i] = 1
        elif(x[i] > 0):
            y[i] = 0
    return(y)


def initial_condition2(x):
    f = np.zeros_like(x)
    x_left = 0.25
    x_right = 0.75
    xm = (x_right - x_left) / 2.0
    f = where((x > x_left) & (x < x_right), np.sin(np.pi * (x - x_left) / (x_right - x_left)) ** 4, f)
    return f

def initial_condition3(x):
    return pow(np.sin(x),2)


if __name__ == '__main__':
    A = 1
    h = 0.01
    k = 0.005
    xbounds = (0, 2)

    lax_wendroff(A, initial_condition3, h, k, xbounds)
