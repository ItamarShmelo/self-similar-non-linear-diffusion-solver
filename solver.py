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
    def dVdZ(self, Z, V_arr, delta:float):
        V = V_arr[0]

        numer = Z*(2.*delta-1.)
        numer += self.m*(self.n+1.)*V*Z 
        numer += self.m*(delta-V)*V

        denom = self.m*Z*(2.*Z + self.m*V)

        return [numer / denom]
