import numpy as np
import math
import pandas as pd


class myRLS:
    def __init__(self, num_vars, lam):
        '''
        num_vars: number of variables including constant
        lam: forgetting factor, usually very close to 1.
        '''
        self.num_vars = num_vars
        self.lam = lam

        # delta controls the initial state.
        self.P = np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1], 1)

        # Variables needed for add_obs
        self.lam_inv = lam ** (-1)
        self.sqrt_lam_inv = math.sqrt(self.lam_inv)

        # A priori error
        self.a_priori_error = 0

        # Count of number of observations added
        self.num_obs = 0

    def add_obs(self, x, t):
        '''
            Add the observation x with label t.
            x is a column vector as a numpy matrix
            t is a real scalar
            '''
        kn= self.P * x
        kd= 1 + (x.T * self.P * x)
        kd= self.lam + (x.T * self.P * x)

        k= kn/kd
        pn = self.P * x * x.T * self.P
        pd =  1 + x.T * self.P * x

        self.P = (self.P - (pn/pd))* self.lam_inv


        self.a_priori_error = float(t - self.w.T * x)
        self.w = self.w +  k *(self.a_priori_error)
        self.num_obs += 1

    def get_error(self):
        '''
        Finds the a priori (instantaneous) error.
        oes not calculate the cumulative effect
        of round-off errors.
        '''
        return self.a_priori_error
