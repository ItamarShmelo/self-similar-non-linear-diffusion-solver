import sys
import os
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOLVER")

EVENT_OCCURED = 1
EPSILON = sys.float_info.epsilon

class Units:
    sigma = 5.670374419e-5

class Solver:
    def __init__(self, *, 
                 geometry=None,  # 'spherical', 'cylindrical', 'planar'
                 rho0=None,      # spatial density coeff
                 omega=None,     # spatial density power
                 g=None,         # opacity coeff
                 alpha=None,     # opacity temmperature power
                 lambdap=None,   # opacity density power
                 f=None,         # energy coeff
                 beta=None,      # energy temmperature power
                 mu=None,        # energy density power
                 r_front=1.,
                 t_front=-1.,
                 A=1.,
                 n=None,
                 m=None,
                 Z0
                 ):
        
        self.init_input = None
        if {geometry, rho0, g, alpha, lambdap, f, beta, mu} != {None}:
            assert None not in {geometry, rho0, omega, g, alpha, lambdap, f, beta, mu}
            self.init_input = {
                'geometry' : geometry, 
                'rho0' : rho0,
                'omega' : omega,
                'g' : g,
                'alpha' : alpha,
                'lambdap' : lambdap,
                'f' : f,
                'beta' : beta,
                'mu' : mu
            }

            m = omega*(1.-mu)
            k = omega*(1+lambdap)
            
            self.A = 16.*Units.sigma*g/(3.*beta*f**((4.+alpha)/beta)*rho0**(lambdap+1.+(1.-mu)*(alpha+4.)/beta))

            if geometry == "spherical":     d=3
            elif geometry == "cylindrical": d=2
            elif geometry == "planar":      d=1
            self.dim = d
            self.n = -(m-d+1.)
            self.m = (4. + alpha - beta)/beta
            self.omega = -(k+m)
        else:
            assert None not in {n, m, omega, A}
            self.n = n
            self.m = m
            self.omega=omega
            self.A = A

        self.Z0 = Z0
        self.ell = -1./(2.+self.omega)
        self.fac = self.A**self.ell

        assert self.n >= 0
        assert self.m >= 1
        assert self.omega > -0.7 and self.omega <= 29.
        assert Z0 < 0.

        self.beta = (2.0 + self.omega) + self.m * (self.n + 1.0)
        self.b = 1.0 # sets the front at x=1 for t=-1

        logger.info(f"n={self.n}")
        logger.info(f"m={self.m}")
        logger.info(f"omega={self.omega}")
        logger.info(f"Z0={self.Z0}")

        self.max_delta = 1.
        self.min_delta = 1. / (2. + self.omega)

        self.delta = None
        self.Z_negative_time = None
        self.V_negative_time = None
        self.max_eta_negative_time = None

        self.event_lambda = lambda Z, V_arr, delta: zero_slope_event(Z, V_arr, m=self.m, omega=self.omega)
        self.event_lambda.terminal = True

        self.dirfigs = f"n_{self.n:g}_m_{self.m:g}_omega={self.omega:g}"
        
        self.calc_delta(self.Z0)
        self.create_interpolation_functions(self.Z0)

    def calc_delta(self, Z0, delta_initial_guess=None):
        """
        Calculate the delta as the root of f
        """
        assert Z0 < 0.

        if delta_initial_guess is None:
            delta_initial_guess = 0.5*(self.max_delta + self.min_delta)

        assert delta_initial_guess < self.max_delta and delta_initial_guess > self.min_delta, f"delta initial guess = {delta_initial_guess:g}"
        self.delta = root(self.f, x0=[delta_initial_guess], args=Z0, tol=1e-8).x[0]
        return self


    def f(self, delta_arr, Z0):
        """
        Calculates the difference between the end points of the integration from A and O
        """
        delta = delta_arr[0]
        if delta >= self.max_delta or delta <= self.min_delta: return [1.]

        sol_A = self.integrate_from_A(delta, Z0)
        sol_O = self.integrate_from_O(delta, Z0)

        if sol_A.status == EVENT_OCCURED and sol_O.status == EVENT_OCCURED:
            return [sol_A.t_events[0][0] - sol_O.t_events[0][0]]
        
        return [sol_A.t[-1] - sol_O.t[-1]]
    
    def integrate_from_A(self, delta, Z0, dense_output=False):
        """
        Integration from A=(0, delta) until point with infinite slope
        """
        V0 = delta 
        V0 += (self.beta*delta-1.) * Z0 / (self.m*(self.m+1.))

        Zmax = -self.m*delta

        solution = solve_ivp(self.dVdZ, t_span=(Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution

    def integrate_from_O(self, delta:float, Z0:float, dense_output:bool=False):
        """
        Integration from O=(0, 0) until point with infinite slope
        """
        V0 = - (2.*delta - 1.) / (self.m*delta) * Z0
        
        alpha = -((2.+self.omega)*delta - 1.) / (self.m*delta)
        gamma = -alpha*(self.beta-1.-self.m)/(self.m*delta**2.)
        
        V0 = alpha*Z0 + gamma*Z0**2
        
        Zmax = -self.m*delta

        solution = solve_ivp(self.dVdZ, t_span=(Z0, Zmax), y0=[V0], args=[delta], events=self.event_lambda, rtol=1e-8, method='LSODA', dense_output=dense_output)
        
        return solution
    
    def dVdZ(self, Z, V_arr, delta:float):
        V = V_arr[0]

        numer = Z*((2.+self.omega)*delta-1.)
        numer += self.m*(self.n+1.)*V*Z 
        numer += self.m*(delta-V)*V

        denom = self.m*Z*((2.+self.omega)*Z + self.m*V)

        return [numer / denom]
    
    def dln_eta_dZ(self, Z, ln_eta_arr, V_Z):
        """
        dln_eta/dZ
        V_Z is a function V(Z)
        """
        return [-1.0/((2.+self.omega)*Z + self.m*V_Z(Z))]

    def create_interpolation_functions(self, Z0:float):
        """
        Creates the functions V(ln_eta), Z(ln_eta)
        """
        assert self.delta is not None
        assert Z0 < 0.0
        
        # integrate from A and O
        sol_V_Z_A = self.integrate_from_A(self.delta, Z0, dense_output=True)
        sol_V_Z_O = self.integrate_from_O(self.delta, Z0, dense_output=True)

        assert sol_V_Z_A.status == EVENT_OCCURED and sol_V_Z_O.status == EVENT_OCCURED

        # create V(Z) from A and from O, since V(Z) is not a function
        # notice integration is from 0.0 since the integration sol_V_Z_A is from Z0
        V_Z_from_A = interp1d([0.0, *sol_V_Z_A.t], [self.delta, *sol_V_Z_A.y[0]], kind='linear', bounds_error=False, fill_value=self.delta)
        V_Z_from_O = interp1d(sol_V_Z_O.t, sol_V_Z_O.y[0], kind='linear', bounds_error=True)
        
        # end Z of integration from A
        Zend_A = sol_V_Z_A.t_events[0][0]
        
        # we split the ln_eta(Z) integration too because we give the functions V_Z_from_A, V_Z_from_O as args
        sol_ln_eta_Z_A = solve_ivp(self.dln_eta_dZ, t_span=(0., Zend_A), y0=[0.0], args=[V_Z_from_A], method='LSODA', rtol=1e-12, atol=1e-8)

        # starting from close to the point of infinte slope (else we have infinite slope and an error)
        Zstart_O = sol_V_Z_O.t_events[0][0]*(1.- 1e3*EPSILON)
        ln_eta_end_A_start_O = sol_ln_eta_Z_A.y[0][-1]
        
        sol_ln_eta_Z_O = solve_ivp(self.dln_eta_dZ, t_span=(Zstart_O, Z0), y0=[ln_eta_end_A_start_O], args=[V_Z_from_O], method='LSODA', rtol=1e-12, atol=1e-8)

        # sol_ln_eta_Z_A end at the same ln_eta that sol_ln_eta_Z_O starts 
        # Z(ln_eta)
        Z_ln_eta = interp1d(np.append(sol_ln_eta_Z_A.y[0][:-1], sol_ln_eta_Z_O.y[0]), np.append(sol_ln_eta_Z_A.t[:-1], sol_ln_eta_Z_O.t), kind='linear', bounds_error=True)

        ln_eta_end = sol_ln_eta_Z_O.y[0][-1]

        ln_eta_from_A_grid = np.linspace(0.0, ln_eta_end_A_start_O, int(1e4), endpoint=False)
        ln_eta_from_O_grid = np.linspace(ln_eta_end_A_start_O, ln_eta_end, int(1e4))

        V_on_grid = np.append(V_Z_from_A(Z_ln_eta(ln_eta_from_A_grid)), V_Z_from_O(Z_ln_eta(ln_eta_from_O_grid)))

        # V(ln_eta) = V(Z(ln_eta))
        V_ln_eta = interp1d(np.append(ln_eta_from_A_grid, ln_eta_from_O_grid), V_on_grid, kind='linear', bounds_error=True)

        self.V_negative_time = V_ln_eta
        self.Z_negative_time = Z_ln_eta

        # maximal eta we can solve
        self.max_eta_negative_time = np.exp(ln_eta_end)
    
    def solve(self, r:np.ndarray, t:float):
        assert self.delta is not None
        assert (self.V_negative_time is not None) and (self.Z_negative_time is not None) and self.max_eta_negative_time is not None

        if t < 0.0:
            eta = r / np.abs(t)**self.delta
            
            Z = np.zeros_like(eta)
            V = np.zeros_like(eta)

            eta_before_front = eta[eta < 1.]

            if any(eta > self.max_eta_negative_time):
                max_r = self.max_eta_negative_time * np.abs(t)**self.delta
                logger.warning(f"Some points are outside the limits, solving from r=0.0 up to r={max_r:g}")
                logger.warning(f"RETURNING NaN FOR THE UNSOLVED r's")

            eta_after_front = eta[np.logical_and(eta >= 1., eta <= self.max_eta_negative_time)]

            ln_eta_after_front = np.log(eta_after_front)

            start_index = len(eta_before_front)
            end_index = start_index+len(eta_after_front)
            
            Z[start_index:end_index] = self.Z_negative_time(ln_eta_after_front)
            V[start_index:end_index] = self.V_negative_time(ln_eta_after_front)

            Z[end_index:] = np.nan
            V[end_index:] = np.nan

            u = (r**(2+self.omega)/np.abs(t)*np.abs(Z))**(1./self.m)
            v = r/np.abs(t)*V
            
            return {
                'u' : u,
                'v' : v,
                'Z' : Z,
                'V' : V,
                'i' : end_index
                }


def zero_slope_event(Z, V_arr, m):
    return (2.*Z + m*V_arr[0])
    
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

    