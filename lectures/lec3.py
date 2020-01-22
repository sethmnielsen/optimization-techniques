import numpy
from scipy import optimize as op
from scipy.optimize import NonlinearConstraint

class UnconstrainedOptim:
    def __init__(self):
        self.x1_t = 1e-4
        self.x2_t = 1e-4

    def search_dir(self):
        pass

    def expected_decrease(self):
        pass

    def find_gradient(self):
