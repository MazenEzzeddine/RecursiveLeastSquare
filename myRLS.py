import numpy as np
import math


class myRLS:
    def __init__(self, num_vars, lam):

        self.num_vars = num_vars
        self.lam = lam

        # delta controls the initial state.
        self.P = np.matrix(np.identity(self.num_vars))
        self.w = np.matrix(np.zeros(self.num_vars))
        self.w = self.w.reshape(self.w.shape[1], 1)

        self.lam_inv = lam ** (-1)

        self.a_priori_error = 0

        self.num_obs = 0

    def add_obs(self, x, t):

        kn= self.P * x
        kd= self.lam + (x.T * self.P * x)

        k= kn/kd
        pn = self.P * x * x.T * self.P
        pd =  self.lam  + x.T * self.P * x

        self.P = (self.P - (pn/pd))* self.lam_inv


        self.a_priori_error = float(t - self.w.T * x)
        self.w = self.w +  k *(self.a_priori_error)
        self.num_obs += 1

    def get_error(self):
        return self.a_priori_error
