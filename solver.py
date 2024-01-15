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
    
    def solve(self, r:np.ndarray, t:float, solve_for_Z_V=False):
        assert self.delta is not None

        if t < 0.0:
            eta = r / np.abs(t)**self.delta
            Z = np.zeros_like(eta)
            V = np.zeros_like(eta)

            eta_after_front = eta[eta >=1.]
            eta_before_front = eta[eta < 1.]

            ln_eta = np.log(eta_after_front)

            solution = solve_ivp(self.ode, t_span=(ln_eta[0], ln_eta[-1]), y0=(0.0, self.delta), t_eval=ln_eta, method='LSODA', rtol=1e-12, atol=1e-8)

            Z[len(eta_before_front):], V[len(eta_before_front):] = solution.y[0], solution.y[1]
            if solve_for_Z_V:
                return Z, V
            
            return (r**2/np.abs(t)*np.abs(Z))**(1./self.m), r/np.abs(t)*V
        else:
            eta = r / np.abs(t)**self.delta
            Z = np.zeros_like(eta)
            V = np.zeros_like(eta)
            eta_initial = min(eta[0], 1e-8)
            ln_eta = np.log(eta)
            
            Z0 = 1.0/eta_initial**2.0
            V0 = -(2.*self.delta-1.)/(self.m*(self.n+1.))*(1. - (self.beta*self.delta-1.)/(self.m*(self.n+1.)*(self.n+3.)*Z0))

            solution = solve_ivp(self.ode, t_span=(np.log(eta_initial), ln_eta[-1]), y0=(Z0, V0), t_eval=ln_eta, method='LSODA', rtol=1e-12, atol=1e-8)
            
            Z[:], V[:] = solution.y[0], solution.y[1]
            
            if solve_for_Z_V:
                return Z, V

            return (r**2/np.abs(t)*np.abs(Z))**(1./self.m), r/np.abs(t)*V
    
def zero_slope_event(Z, V_arr, m):
    
if __name__ == "__main__":
    Z0=-1e-5

    solver = Solver(n=2., m=7.0).calc_delta(Z0=Z0)

    
    t=1.
    r=np.linspace(1e-10, 10.0, 100)
    Z, V = solver.solve(r, t, solve_for_Z_V=True)

    plt.plot(r, V, 'o')
    plt.grid()
    # plt.ylim(0.0, 0.8)
    plt.savefig(os.path.join(solver.dirfigs, "V.png"))
    plt.close()

    plt.plot(r, Z, 'o')
    plt.grid()
    # plt.ylim(0., 2.)
    plt.savefig(os.path.join(solver.dirfigs, "Z.png"))
    plt.close()

    plt.plot(Z, V, 'o')
    plt.grid()
    # plt.ylim(0., 2.)
    plt.savefig(os.path.join(solver.dirfigs, "ZV.png"))
    plt.close()




    u, v = solver.solve(r, t)

    
    # for (ri,ui) in zip(r,u):
    #     print(f"r={ri:.4f}, u={ui:.4f}")

    plt.plot(r, u, 'o')
    plt.grid()
    plt.savefig(os.path.join(solver.dirfigs, "u.png"))
    plt.close()

    plt.plot(r, np.abs(v), 'o')
    plt.grid()
    plt.savefig(os.path.join(solver.dirfigs, "v.png"))
    plt.close()

    