import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from matplotlib import pyplot as plt
import os

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SOLVER")
