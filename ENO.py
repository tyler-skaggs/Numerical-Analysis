from pickletools import uint2

import numpy as np
from sympy import *

### This code is based on https://github.com/chairmanmao256/ENO-Linear-advection/blob/main/ENO.py

class ENO:
    '''
    ### Description

    The class to solve the 1-D linear advection equation using ENO
    reconstruction scheme of order 3.
    '''

    def __init__(self, l, r, dx, dt, problem, deriv, init, order=3):
        '''
        ### Description
        Initialize the ENO_advection class with:

        `a`: The convection speed. Default value is 1.0.
        `ni`: The number of internal cells. Default value is 100.
        '''

        self.order = order
        self.ib = order - 1  # k-th order scheme requires k-1 ghost cells
        self.ni = int((r - l) / dx) + 1
        self.im = self.ni + self.ib  # to iterate through all the internal cells: for i in range(ib, im)
        self.f = problem
        self.init = init
        self.delx = dx
        self.dt = dt

        self.u = np.zeros(2 * self.ib + self.ni)  # solution vector
        self.u_m = np.zeros_like(self.u)  # stores the middle step (previous solution) in Runge-Kutta scheme

        ib = self.ib
        self.xc = np.linspace(l - ib * self.delx, r + ib * self.delx, 2 * self.ib + self.ni)

        self.a = max(abs(deriv(self.u)))

        # interface values
        self.ul = np.zeros(2 * self.ib + self.ni + 1)
        self.ur = np.zeros(2 * self.ib + self.ni + 1)
        self.flux = np.zeros(2 * self.ib + self.ni + 1)

        # Newton's devided difference
        self.V = np.zeros((2 * self.ib + self.ni + 1, 3))
        self.global_t = 0.0

    def set_initial(self):
        '''
        ### Description

        Set the initial condition
        '''
        self.u = self.init(self.xc)

    def set_ghost(self):
        '''
        ### Description

        Set the value in the boundary ghost cells based on a periodic initial condition
        '''
        # left boundary
        for k in range(1, self.order):
            self.u[self.ib - k] = self.u[self.im - k]

        # right boundary
        for k in range(self.order-1):
            self.u[self.im + k] = self.u[self.ib + k]

    def NDD(self):
        '''
        ### Description

        Compute the Newton Divde Difference of the function value array.
        Function value array is stored on the cell centers.

            +--------+
            |        |
        i> |  u[i]  | <i+1
            |        |
            +--------+

        ib is the starting index of the internal cells. It also equals to
        the ghost cells used.

        We only compute the the Diveded Difference to the order 3:
        V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}, x_{i+5/2}]

        We also assume uniform cell length: \Delta x = const.
        '''
        nn = self.u.shape[0]  # total number of cells, including the ghost cells

        self.V = np.zeros((nn + 1, self.order))
        self.V[0:nn, 0] = self.u.copy()  # order-1: V[x_{i-1/2}, x_{i+1/2}]

        for k in range(1, self.order):
            self.V[0:nn - k, k] = self.V[1:nn - (k - 1), k - 1] - self.V[0:nn - k, k - 1]  # order-(k+1): V[x_{i-1/2}, x_{i+1/2},..., x_{i+k+1/2}]
        # self.V[0:nn-2, 2] = self.V[1:nn-1, 1] - self.V[0:nn-2, 1]      # order-3: V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}, x_{i+5/2}]

    def ENO_weight(self, r: int):
        '''
        ### Description:

        Compute the ENO weight based on the left shift of the stencil
        '''
        crj = np.zeros(self.order)
        for j in range(self.order):
            for m in range(j + 1, self.order + 1):
                de = 1.0
                no = 0.0
                for l in range(self.order + 1):
                    if l != m:
                        de = de * (m - l)

                for l in range(self.order + 1):
                    if l != m:
                        ee = 1.0
                        for q in range(self.order + 1):
                            if q != m and q != l:
                                ee *= (r - q + 1)
                        no += ee

                crj[j] += float(no) / float(de)

        return crj

    def ENO_reconstruction(self):
        '''
        ### Description:

        Perform ENO reconstruction cell-wise
        '''
        ib, im = self.ib, self.im
        # compute the NDD first
        self.NDD()

        # reconstruct on internal cell faces, cell by cell
        for i in range(ib, im):
            # initial stencil
            stencil = np.array([i, i + 1])
            for k in range(self.order - 1):  # the number of interfaces in the stencil: k+2
                L, R = stencil[0], stencil[-1]

                # determine the expanded stencil by evaluating NDD
                stencilL = np.append(L - 1, stencil)
                stencilR = np.append(stencil, R + 1)

                V2L = self.V[stencilL[0], k + 1]  # note that subscript k+1 retrives order k+2 diveded difference
                V2R = self.V[stencilR[0], k + 1]

                if abs(V2L) < abs(V2R):
                    stencil = stencilL.copy()
                else:
                    stencil = stencilR.copy()

            # final stencil is now stored in `stencil`. Evaluate the stencil shift.
            '''
            +-------------+-------------+------------+
            |             |< i          |            |
            |< stencil[0] |< stencil[1] |< stencil[2]|
            |             |             |            |
            +-------------+-------------+------------+
             ^center used  ^center used   ^center used

            The plot above is an example of 3rd-order stencil, where the left shift r = 1.
            '''
            r = i - stencil[0]

            # obtain the ENO weight
            cL = self.ENO_weight(r)
            cR = self.ENO_weight(r - 1)

            # obtain the cell-center values
            vv = self.u[stencil[0:-1]]

            self.ul[i + 1] = cL @ vv
            self.ur[i] = cR @ vv

        # set the boundary state by using periodic condition
        self.ul[ib] = self.ENO_weight(-1) @ self.u[ib-1:ib + 2]
        self.ur[im] = self.ENO_weight(0) @ self.u[im - 1: im+2]


    def LAX_flux(self):
        '''
        ### Description

        Compute the L-F flux based on the reconstructed values
        '''
        self.flux = 1/2 * (self.f(self.ul) + self.f(self.ur) - self.a * (self.ur - self.ul))

    def Runge_Kutta(self):
        self.set_ghost()
        self.u_m = self.u.copy()

        alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
        alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
        alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]

        for j in range(3):
            self.ENO_reconstruction()
            self.LAX_flux()
            self.u[self.ib:self.im] = alpha1[j] * self.u_m[self.ib:self.im] + alpha2[j] * self.u[self.ib:self.im] - \
                                      alpha3[j] * self.dt / self.delx * (
                                                  self.flux[self.ib + 1:self.im + 1] - self.flux[self.ib:self.im])

        self.global_t += self.dt
        return self.dt

class EENO:
    '''
    ### Description

    The class to solve the 1-D linear advection equation using ENO
    reconstruction scheme of order 3.
    '''

    def __init__(self, l, r, dx, dt, problem, deriv, init, order=3):
        '''
        ### Description
        Initialize the ENO_advection class with:

        `a`: The convection speed. Default value is 1.0.
        `ni`: The number of internal cells. Default value is 100.
        '''

        self.order = order
        self.ib = 5  # k-th order scheme requires k+2 ghost cells
        self.ni = int((r - l) / dx) + 1
        self.im = self.ni + self.ib  # to iterate through all the internal cells: for i in range(ib, im)
        self.f = problem
        self.init = init
        self.dx = dx
        self.dt = dt
        self.m = 2

        self.u = np.zeros(2 * self.ib + self.ni)  # solution vector
        self.u_m = np.zeros_like(self.u)  # stores the middle step (previous solution) in Runge-Kutta scheme

        ib = self.ib
        self.xc = np.linspace(l - ib * self.dx, r + ib * self.dx, 2 * self.ib + self.ni)

        self.alpha = max(abs(deriv(self.u)))

        # interface values
        self.p_plus = np.zeros(2 * self.ib + self.ni + 1)
        self.p_minus = np.zeros(2 * self.ib + self.ni + 1)
        self.f_hat = np.zeros(2 * self.ib + self.ni + 1)
        self.flux = np.zeros(self.ni)

        # Newton's divided difference
        self.VPlus = np.zeros((2 * self.ib + self.ni + 1, 2*self.m+2))
        self.VMinus = np.zeros((2 * self.ib + self.ni + 1, 2*self.m+2))

        self.global_t = 0.0

    def set_initial(self):
        '''
        ### Description

        Set the initial condition
        '''
        self.u = self.init(self.xc)

    def set_ghost(self):
        '''
        ### Description

        Set the value in the boundary ghost cells based on a periodic initial condition
        '''
        # left boundary
        for k in range(1, self.order):
            self.u[self.ib - k] = self.u[self.im - k-1]

        # right boundary
        for k in range(self.order):
            self.u[self.im + k] = self.u[self.ib + k+1]

    def NDD(self):
        '''
        ### Description

        Compute the Newton Divde Difference of the function value array.
        Function value array is stored on the cell centers.

            +--------+
            |        |
        i> |  u[i]  | <i+1
            |        |
            +--------+

        ib is the starting index of the internal cells. It also equals to
        the ghost cells used.

        We only compute the Diveded Difference to the order 3:
        V[x_{i-1/2}, x_{i+1/2}, x_{i+3/2}, x_{i+5/2}]

        We also assume uniform cell length: \Delta x = const.
        '''
        nn = self.u.shape[0]  # total number of cells, including the ghost cells

        self.VPlus = np.zeros((nn + 1, 2*self.m+2))
        self.VPlus[0:nn, 0] = 1/2*(self.f(self.u.copy()) + self.alpha * self.u.copy())

        self.VMinus = np.zeros((nn + 1, 2*self.m+2))
        self.VMinus[0:nn, 0] = 1/2*(self.f(self.u.copy()) - self.alpha * self.u.copy())

        for k in range(1, 2 * self.m + 1):
            self.VPlus[0:nn - k, k] = (self.VPlus[1:nn - (k - 1), k - 1] - self.VPlus[0:nn - k, k - 1]) / (k*self.dx)
            self.VMinus[0:nn - k, k] = (self.VMinus[1:nn - (k - 1), k - 1] - self.VMinus[0:nn - k, k - 1]) / (k*self.dx)
            # order-(k+1): V[x_{i-1/2}, x_{i+1/2},..., x_{i+k+1/2}]

    def ENO_reconstruction(self):
        '''
        ### Description:

        Perform ENO reconstruction cell-wise
        '''
        ib, im = self.ib, self.im
        # compute the NDD first
        self.NDD()

        # reconstruct on internal cell faces, cell by cell
        for j in range(ib-1, im+1):
            # initial stencil
            k_P = np.array([j])
            k_M = np.array([j+1])

            Q_Plus = 1/2 * (self.f(self.u[j]) + self.alpha * self.u[j])
            Q_Minus = 1/2 * (self.f(self.u[j+1]) - self.alpha * self.u[j+1])

            for n in range(2*self.m+1):  # the number of interfaces in the stencil
                ### For Q_plus
                a = self.VPlus[k_P[0], n + 1]  # note that subscript n+1 retrives order n+2 divided difference
                b = self.VPlus[k_P[0]-1, n + 1]
                prod = 1

                if abs(a) < abs(b):
                    c = a
                    for k in k_P:
                        prod *= (self.xc[j] - self.xc[k] + self.dx/2)
                    k_P = np.append(k_P, k_P[-1] + 1)
                else:
                    c = b
                    for k in k_P:
                        prod *= (self.xc[j] - self.xc[k] + self.dx/2)
                    k_P = np.append(k_P[0]-1, k_P)

                Q_Plus += c * prod

                ### For Q_minus
                a = self.VMinus[k_M[0], n + 1]  # note that subscript n+1 retrives order n+2 divided difference
                b = self.VMinus[k_M[0] - 1, n + 1]
                prod = 1

                if abs(a) < abs(b):
                    c = a
                    for k in k_M:
                        prod *= (self.xc[j] - self.xc[k] + self.dx/2)
                    k_M = np.append(k_M, k_M[-1] + 1)
                else:
                    c = b
                    for k in k_M:
                        prod *= (self.xc[j] - self.xc[k] + self.dx/2)
                    k_M = np.append(k_M[0] - 1, k_M)

                Q_Minus += c * prod

            self.p_plus[j] = Q_Plus
            self.p_minus[j] = Q_Minus

            self.f_hat[j] = Q_Plus + Q_Minus

        # set the boundary state by using periodic condition
        #self.f_hat[ib-1] = self.f_hat[im-1]


    def approx_flux(self):
        '''
        ### Description

        Compute the L-F flux based on the reconstructed values
        '''
        self.flux = self.f_hat[self.ib:self.im] - self.f_hat[(self.ib-1):(self.im-1)]


    def Runge_Kutta(self):
        self.u_m = self.u.copy()

        alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
        alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
        alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]

        for j in range(3):
            self.set_ghost()
            self.ENO_reconstruction()
            self.approx_flux()
            self.u[self.ib:self.im] = alpha1[j] * self.u_m[self.ib:self.im] + alpha2[j] * self.u[self.ib:self.im] - \
                                      alpha3[j] * self.dt / self.dx * self.flux

        self.global_t += self.dt
        return self.dt

class WENO:
    '''
    ### Description

    The class to solve the 1-D linear advection equation using ENO
    reconstruction scheme of order 3.
    '''

    def __init__(self, l, r, dx, dt, problem, deriv, init, order=3, rr = 2):
        '''
        ### Description
        Initialize the ENO_advection class with:

        `a`: The convection speed. Default value is 1.0.
        `ni`: The number of internal cells. Default value is 100.
        '''
        self.order = order
        self.r = rr
        self.ib = order   # k-th order scheme requires k-1 ghost cells
        self.ni = int((r - l) / dx) + 1
        self.im = self.ni + self.ib  # to iterate through all the internal cells: for i in range(ib, im)
        self.f = problem
        self.fprime = deriv
        self.init = init
        self.dx = dx
        self.dt = dt

        self.u = np.zeros(2 * self.ib + self.ni)  # solution vector
        self.u_m = np.zeros_like(self.u)  # stores the middle step (previous solution) in Runge-Kutta scheme

        ib = self.ib
        self.xc = np.linspace(l - ib * self.dx, r + ib * self.dx, 2 * self.ib + self.ni)

        self.alpha = max(abs(deriv(self.u)))

        # interface values
        self.L = np.zeros(2 * self.ib + self.ni)
        self.Rl = np.zeros(2 * self.ib + self.ni)
        self.Rr =np.zeros(2 * self.ib + self.ni)

    def set_initial(self):
        '''
        ### Description

        Set the initial condition
        '''
        self.u = self.init(self.xc)


    def ENO_weight(self, r: int):
        '''
        ### Description:

        Compute the ENO weight based on the left shift of the stencil
        '''
        crj = np.zeros(self.order)
        for j in range(self.order):
            for m in range(j + 1, self.order + 1):
                de = 1.0
                no = 0.0
                for l in range(self.order + 1):
                    if l != m:
                        de = de * (m - l)

                for l in range(self.order + 1):
                    if l != m:
                        ee = 1.0
                        for q in range(self.order + 1):
                            if q != m and q != l:
                                ee *= (r - q + 1)
                        no += ee

                crj[j] += float(no) / float(de)

        return crj


    def set_ghost(self):
        '''
        ### Description

        Set the value in the boundary ghost cells based on a periodic initial condition
        '''
        # left boundary
        for k in range(1, self.order):
            self.u[self.ib - k] = self.u[self.im - k-1]

        # right boundary
        for k in range(self.order):
            self.u[self.im + k] = self.u[self.ib + k+1]


    def ENO_reconstruction(self):
        '''
        ### Description:

        Perform ENO reconstruction cell-wise
        '''
        ib, im = self.ib, self.im
        # compute the NDD first
        #self.NDD()

        # reconstruct on internal cell faces, cell by cell
        for j in range(ib-1, im+1):
             def R(x):
                 e = pow(10,-5)

                 if self.r == 2:
                     IS_j = pow(self.u[j] - self.u[j-1], 2)
                     IS_jp = pow(self.u[j+1] - self.u[j], 2)

                     if self.fprime(self.u[j]) > 0:
                         a0 = 1/(2 * pow(e + IS_j, 2))
                         a1 = 1/pow(e + IS_jp, 2)
                     else:
                         a0 = 1/pow(e + IS_j, 2)
                         a1 = 1 / (2 * pow(e + IS_jp, 2))

                     v0 = self.u[[j-1, j, j+1]]
                     v1 = self.u[[j, j + 1, j+2]]

                     val = (a0 / (a0 + a1) * self.ENO_weight(1) @ v0 +
                            a1 / (a0 + a1) * self.ENO_weight(0) @ v1)

                 elif self.r == 3:
                     IS_j = (pow( self.u[j-1] - self.u[j-2], 2) + pow(self.u[j] - self.u[j-1], 2) +
                             pow((self.u[j] - 2 * self.u[j-1] - self.u[j-2]), 2))
                     IS_jp = (pow( self.u[j] - self.u[j-1], 2) + pow(self.u[j+1] - self.u[j], 2) +
                              pow((self.u[j+1] - 2 *self.u[j] + self.u[j-1]), 2))
                     IS_jpp = (pow(self.u[j+1] - self.u[j], 2) + pow(self.u[j + 2] - self.u[j+1], 2) +
                               pow((self.u[j + 2] - 2 * self.u[j+1] + self.u[j]), 2))

                     if self.fprime(self.u[j]) > 0:
                         a0 = 1/(12 * pow(e + IS_j, 3))
                         a1 = 1/(2 * pow(e + IS_jp, 3))
                         a2 = 1/(4 * pow(e + IS_jpp, 3))
                     else:
                         a0 = 1 / (4 * pow(e + IS_j, 3))
                         a1 = 1 / (2 * pow(e + IS_jp, 3))
                         a2 = 1 / (12 * pow(e + IS_jpp, 3))

                     v0 = self.u[[j - 2, j - 1, j, j + 1]]
                     v1 = self.u[[j - 1, j, j + 1, j + 2]]
                     v2 = self.u[[j, j + 1, j + 2, j + 3]]

                     val = (a0 / (a0 + a1 + a2) * self.ENO_weight(2) @ v0 +
                            a1 / (a0 + a1 + a2) * self.ENO_weight(1) @ v1 +
                            a2 / (a0 + a1 + a2) * self.ENO_weight(0) @ v2)

                 return val


             self.Rr[j] = R(self.xc[j] + self.dx/2)
             self.Rl[j] = R(self.xc[j] - self.dx/2)


    def setL(self):
        '''
        ### Description

        Compute the L-F flux based on the reconstructed values
        '''
        def flux(a, b):
            return 1/2 * (self.f(a) + self.f(b) - self.alpha * (b - a))

        self.L[self.ib:self.im] = -1 / self.dx * (flux(self.Rr[self.ib:self.im], self.Rr[self.ib+1:self.im+1]) -
                                                  flux(self.Rl[self.ib-1:self.im-1], self.Rl[self.ib:self.im]))


    def Runge_Kutta(self):
        self.u_m = self.u.copy()

        alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
        alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
        alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]

        for j in range(3):
            self.set_ghost()
            self.ENO_reconstruction()
            self.setL()
            self.u[self.ib:self.im] = alpha1[j] * self.u_m[self.ib:self.im] + alpha2[j] * self.u[self.ib:self.im] + \
                                      alpha3[j] * self.dt * self.L[self.ib:self.im]

        return self.dt

class EWENO:
    '''
    ### Description

    The class to solve the 1-D linear advection equation using ENO
    reconstruction scheme of order 3.
    '''

    def __init__(self, l, r, dx, dt, problem, deriv, init, rr = 3):
        '''
        ### Description
        Initialize the ENO_advection class with:
        '''

        self.order = rr
        self.r = rr
        self.ib = rr   # number of extra boundary cells needed
        self.ni = int((r - l) / dx) + 1
        self.im = self.ni + self.ib  # to iterate through all the internal cells: for i in range(ib, im)
        self.f = problem
        self.fprime = deriv
        self.init = init
        self.dx = dx
        self.dt = dt

        self.u = np.zeros(2 * self.ib + self.ni)  # solution vector
        self.u_m = np.zeros_like(self.u)  # stores the middle step (previous solution) in Runge-Kutta scheme

        ib = self.ib
        self.xc = np.linspace(l - ib * self.dx, r + ib * self.dx, 2 * self.ib + self.ni)

        self.alpha = max(abs(deriv(self.u)))

        # interface values
        self.L = np.zeros(2 * self.ib + self.ni)
        self.f_plus = np.zeros(2 * self.ib + self.ni)
        self.f_minus = np.zeros(2 * self.ib + self.ni)

    def set_initial(self):
        '''
        ### Description

        Set the initial condition
        '''
        self.u = self.init(self.xc)

    def set_ghost(self):
        '''
        ### Description

        Set the value in the boundary ghost cells based on a periodic initial condition
        '''
        # left boundary
        for k in range(1, self.order):
            self.u[self.ib - k] = 0#self.u[self.im - k-1]

        # right boundary
        for k in range(self.order):
            self.u[self.im + k] = 1#self.u[self.ib + k+1]


    def ENO_reconstruction(self):
        '''
        ### Description:

        Perform ENO reconstruction cell-wise
        '''
        ib, im = self.ib, self.im
        # compute the NDD first
        #self.NDD()

        # reconstruct on internal cell faces, cell by cell
        a = np.array([[[-1 / 2, 3 / 2, 0], [1 / 2, 1 / 2, 0], [0, 0, 0]],
                      [[1 / 3, -7 / 6, 11 / 6], [-1 / 6, 5 / 6, 1 / 3], [1 / 3, 5 / 6, -1 / 6]]])

        C = np.array([[1 / 3, 2 / 3, 0],
                      [1 / 10, 6 / 10, 3 / 10]])

        p = self.r
        e = pow(10, -6)

        def q(k, r, g):
            sum = 0
            for l in range(r):
                sum += a[r - 2, k, l] * g[l]
            return sum

        fl_plus = 1/2 * (self.f(self.u) + self.alpha * self.u)
        fl_minus = 1/2 * (self.f(self.u) - self.alpha * self.u)

        for j in range(ib-1, im+1):

            ### Positive
            if self.r == 3:
                IS_k = np.array([ 13/12 * pow(fl_plus[j-2] - 2 * fl_plus[j-1] + fl_plus[j], 2) + 1/4 *
                                  pow(fl_plus[j-1] - 4 * fl_plus[j-1] + 3 * fl_plus[j], 2),

                                  13/12 * pow(fl_plus[j-1] - 2 * fl_plus[j] + fl_plus[j+1], 2) + 1/4 *
                                  pow(fl_plus[j-1] - fl_plus[j+1], 2),

                                  13/12 * pow(fl_plus[j] - 2 * fl_plus[j+1] + fl_plus[j+2], 2) + 1/4 *
                                  pow(3 * fl_plus[j] - 4 * fl_plus[j+1] + fl_plus[j+1], 2)])
            else:
                IS_k = np.array([pow(fl_plus[j] - fl_plus[j-1], 2),
                                 pow(fl_plus[j+1] - fl_plus[j], 2), 0])

            alpha = C[self.r-2, :] / pow(e + IS_k, p)

            omega = alpha / np.sum(alpha)

            sum = 0
            for k in range(self.r):
                sum += omega[k] * q(k, self.r, fl_plus[j+k-self.r + 1:j+k+1])

            self.f_plus[j] = sum

            ### Negative
            if self.r == 3:
                IS_k = np.array([13 / 12 * pow(fl_minus[j - 2] - 2 * fl_minus[j - 1] + fl_minus[j], 2) + 1 / 4 *
                                 pow(fl_minus[j - 1] - 4 * fl_minus[j - 1] + 3 * fl_minus[j], 2),

                                 13 / 12 * pow(fl_minus[j - 1] - 2 * fl_minus[j] + fl_minus[j + 1], 2) + 1 / 4 *
                                 pow(fl_minus[j - 1] - fl_minus[j + 1], 2),

                                 13 / 12 * pow(fl_minus[j] - 2 * fl_minus[j + 1] + fl_minus[j + 2], 2) + 1 / 4 *
                                 pow(3 * fl_minus[j] - 4 * fl_minus[j + 1] + fl_minus[j + 1], 2)])
            else:
                IS_k = np.array([pow(fl_minus[j] - fl_minus[j - 1], 2),
                                 pow(fl_minus[j + 1] - fl_minus[j], 2),
                                 0])

            alpha = C[self.r - 2, :] / pow(e + IS_k, p)

            omega = alpha / np.sum(alpha)

            sum = 0
            for k in range(self.r):
                sum += omega[k] * q(k, self.r, fl_minus[j + k - self.r + 1:j + k+1])

            self.f_minus[j] = sum


    def setL(self):
        '''
        ### Description

        Compute the L-F flux based on the reconstructed values
        '''
        f_Half = self.f_plus + self.f_minus

        self.L[self.ib:self.im] = -1/self.dx * (f_Half[self.ib:self.im] - f_Half[self.ib-1:self.im-1])


    def Runge_Kutta(self):
        if self.r == 2:
            self.u_m = self.u.copy()
            alpha1 = [1.0, 3.0 / 4.0, 1.0 / 3.0]
            alpha2 = [0.0, 1.0 / 4.0, 2.0 / 3.0]
            alpha3 = [1.0, 1.0 / 4.0, 2.0 / 3.0]

            for j in range(3):
                self.set_ghost()
                self.ENO_reconstruction()
                self.setL()
                self.u[self.ib:self.im] = (alpha1[j] * self.u_m[self.ib:self.im] +
                                           alpha2[j] * self.u[self.ib:self.im] +
                                           alpha3[j] * self.dt * self.L[self.ib:self.im])

        else:
            uk = [self.u.copy(), self.u.copy(), self.u.copy(), self.u.copy(), self.u.copy()]
            Lk = [self.u.copy(), self.u.copy(), self.u.copy(), self.u.copy()]

            alpha0 = [1, 1, 1, -1/3] #[1, 1/2, 1/9, 0]
            alpha1 = [0, 0, 0, 1/3] #[0, 1/2, 2/9, 1/3]
            alpha2 = [0, 0, 0, 2/3] #[0, 0, 2/3, 1/3]
            alpha3 = [0, 0, 0, 1/3] #[0, 0, 0, 1/3]
            alphaL0 = [1/2, 0, 0, 0]#[1/2, -1/4, -1/9, 0]
            alphaL1 = [0, 1/2, 0, 0] #[0, 1/2, -1/3, 1/6]
            alphaL2 = [0, 0, 1, 0] #[0, 0, 1, 0]
            alphaL3 = [0, 0, 0, 1/6] #[0, 0, 0, 1/6]


            for j in range(4):
                self.set_ghost()
                self.ENO_reconstruction()
                self.setL()
                Lk[j] = self.L
                uk[j+1][self.ib:self.im] = \
                    (   alpha0[j] * uk[0][self.ib:self.im] +
                        alpha1[j] * uk[1][self.ib:self.im] +
                        alpha2[j] * uk[2][self.ib:self.im] +
                        alpha3[j] * uk[3][self.ib:self.im] +
                        alphaL0[j] * self.dt * Lk[0][self.ib:self.im] +
                        alphaL1[j] * self.dt * Lk[1][self.ib:self.im] +
                        alphaL2[j] * self.dt * Lk[2][self.ib:self.im] +
                        alphaL3[j] * self.dt * Lk[3][self.ib:self.im])

                self.u[self.ib:self.im] = uk[j+1].copy()[self.ib:self.im]

        return self.dt