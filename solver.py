import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot as plt
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOLVER")

class Solver:
    def __init__(self, *, n:float, m:float):
        self.n = n
        self.m = m

        self.beta = 2.0 + self.m * (self.n + 1.0)
        self.b = 1.0 # sets the front at x=1 for t=-1

        logger.info(f"n={self.n}")
        logger.info(f"m={self.m}")

        self.delta = None
        
        self.delta_temp = None
        
        self.event_lambda = lambda Z, V_arr, delta: zero_slope_event(Z, V_arr, m=self.m)
        self.event_lambda.terminal = True

        self.counter = 0
        self.dirfigs = f"n_{self.n:.2f}_m_{self.m:.2f}"
        os.makedirs(self.dirfigs, exist_ok=True)
        

    def calc_delta(self, Z0, delta_initial_guess=3./4.):
        self.delta = root(self.f, x0=[delta_initial_guess], args=Z0, tol=1e-8).x[0]
        return self


    def f(self, delta_arr, Z0):
        delta = delta_arr[0]
        if delta >= 1. or delta <= 0.5: return 1.

        sol_A = self.integrate_from_A(delta, Z0)
        sol_O = self.integrate_from_O(delta, Z0)

        plt.plot(sol_A.t, sol_A.y[0])
        plt.plot(sol_O.t, sol_O.y[0])

        plt.xlabel("Z")
        plt.ylabel("V")
        plt.grid()
        plt.savefig(os.path.join(self.dirfigs, f"fig_{self.counter}.png"))
        self.counter += 1
        plt.close()
        
        return [sol_A.y[0][-1] - sol_O.y[0][-1]]
    
    def integrate_from_A(self, delta, Z0, dense_output=False):
        V0 = delta 
        V0 += (self.beta*delta-1.) * Z0 / (self.m*(self.m+1.))

        Zmax = -self.m*delta/2.

        solution = solve_ivp(self.dVdZ, t_span=(Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution

    def integrate_from_O(self, delta:float, Z0:float, dense_output:bool=False):
        V0 = 1. - ((2.*delta - 1.)*(3. + 2.*self.m)/(self.m*delta) - 3. + self.n)*Z0
        V0 = 1.- V0*(self.beta*delta-1.)*(self.m+1)/(self.m*delta**2.)*Z0
        V0 = - (2.*delta - 1.) * Z0 / self.m*V0
        Zmax = -self.m*delta/2.


        solution = solve_ivp(self.dVdZ, t_span=(Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution
    
    def dVdZ(self, Z, V_arr, delta:float):
        V = V_arr[0]

        numer = Z*(2.*delta-1.)
        numer += self.m*(self.n+1.)*V*Z 
        numer += self.m*(delta-V)*V

        denom = self.m*Z*(2.*Z + self.m*V)

        return [numer / denom]
    
    def ode(self, ln_eta, y):
        Z, V = y
        dZ_dln_eta = -(2.*Z+self.m*V)

        dV_dln_eta = -(2.*self.delta-1.)
        dV_dln_eta -= self.m*(self.n+1.)*V
        dV_dln_eta -= 0. if np.abs(Z) < np.finfo(float).eps*1024. else self.m*(self.delta-V)*V/Z 
        dV_dln_eta /= self.m
        return [dZ_dln_eta, dV_dln_eta]
def zero_slope_event(Z, V_arr, m):
    return (2.*Z + m*V_arr[0])
