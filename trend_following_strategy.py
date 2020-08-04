from datetime import *

import numpy as np
import pandas as pd

import rqdatac as rq
from rqdatac import *

from trend_following_objects import RegimeInfo, ParameterData


class EstimationParameter(object):
    """
    Estimate parameters, such as lambda, mu, sigma
    """

    def __init__(self, data, price_type, regime_info):
        """
        Initialize the EsitimationPara
        """

        self.data = data
        self.vt_symbol = regime_info.vt_symbol
        self.bull_p = regime_info.bull_profit
        self.bear_l = regime_info.bear_loss

        self.para_info = ParameterData(self.vt_symbol)

        self.regimes = []
        self.date_array = self.data.index.values
        self.close_array = self.data[price_type].values
        

    def __str__(self):
        return f"Estimate Parameters for {self.vt_symbol}"
    
    def get_para_info(self, state):

        # parameter estimation
        self.find_bull_bear(state)
        self.lambda_estimation()


        return self.para_info

    def find_bull_bear(self, state):
        n = len(self.close_array)
        signal = state
        max_s, min_s = self.close_array[0], self.close_array[0]
        self.regimes = []
        for i in range(1, n-1, 1):
            if signal == "bull":
                if (self.close_array[i]/max_s < 1 - self.bear_l):
                    closes, dates = self.close_array[:i], self.date_array[:i]
                    start = dates[closes == min_s][-1]
                    end = dates[closes == max_s][-1]
                    bull_info = {
                        "state": signal,
                        "start_date": start,
                        "end_date": end,
                        "min_point": min_s,
                        "max_point": max_s
                    }
                    self.regimes.append(bull_info)
                    signal = "bear"
                    min_s = self.close_array[i]
                else:
                    max_s = max(self.close_array[i], max_s)
            elif signal == "bear":
                if (self.close_array[i] / min_s > 1 + self.bull_p):
                    closes, dates = self.close_array[:i], self.date_array[:i]
                    start = dates[closes == max_s][-1]
                    end = dates[closes == min_s][-1]
                    bear_info = {
                        "state": signal,
                        "start_date": start,
                        "end_date": end,
                        "min_point": min_s,
                        "max_point": max_s
                    }
                    self.regimes.append(bear_info)
                    signal = "bull"
                    max_s = self.close_array[i]
                else:
                    min_s = min(self.close_array[i], min_s)
        # return self.regimes[1:]

    def lambda_estimation(self):
        bull_t, bear_t = [], []
        for regime in self.regimes:
            t = (pd.to_datetime(regime["end_date"]) - pd.to_datetime(regime["start_date"])).days / 365
            if regime["state"] == "bull":
                bull_t.append(t)
            elif regime["state"] == "bear":
                bear_t.append(t)
            else:
                print("Invalid Market Regime:", regime)
        
        bull_t, bear_t = np.array(bull_t), np.array(bear_t)

        self.para_info.bull_lambda = (1 / bull_t).mean()
        self.para_info.bear_lambda = (1 / bear_t).mean()


    def seperate_bull_bear(self):
        bull_date = []
        bear_date = []
        bull_indice = []
        bear_indice = []
        n = len(regimes)
        for i in range(n):
            state = regimes[i]
            if state[0] == 'bull':
                bull_date.append(state[1])
                bull_indice.append(state[2])
            else:
                bear_date.append(state[1])
                bear_indice.append(state[2])

            if i == n-1:
                if state[0] == 'bull':
                    bear_date.append(state[3])
                    bear_indice.append(state[4])
                else:
                    bull_date.append(state[3])
                    bull_indice.append(state[4])

        bull_date, bear_date = np.array(bull_date), np.array(bear_date)
        bull_indice, bear_indice = np.array(bull_indice), np.array(bear_indice)

        return bull_date, bull_indice, bear_date, bear_indice



    def seperate_bull_bear(self):
        bull_date = []
        bear_date = []
        bull_indice = []
        bear_indice = []
        for regime in self.regimes[1:]:
            if regime["state"] == 'bull':
                bull_date.append()
                bull_indice.append(state[2])
            else:
                bear_date.append(state[1])
                bear_indice.append(state[2])

            if i == n-1:
                if state[0] == 'bull':
                    bear_date.append(state[3])
                    bear_indice.append(state[4])
                else:
                    bull_date.append(state[3])
                    bull_indice.append(state[4])

        bull_date, bear_date = np.array(bull_date), np.array(bear_date)
        bull_indice, bear_indice = np.array(bull_indice), np.array(bear_indice)

        return bull_date, bull_indice, bear_date, bear_indice

    def get_regimes(self):
        return self.regimes




class TrendFollowing(object):

    def __init__(self, market_info, para_info):
        pass


    def projected_sor(A, b, V, n, epsilon, omega, upper, lower):
        """
        Implement Projected SOR method to find buy & sell boundary.
        """

        converged = False
        x0 = V
        x1 = np.zeros(n)
        while not converged:
            for i in range(n):
                if i == 0:
                    ai, ap = A[i, i], A[i, i+1]
                    x_gs = (-ap*x0[i+1] + b[i])/ai
                elif i != (n-1):
                    am, ai, ap = A[i, i-1], A[i, i], A[i, i+1]
                    x_gs = (- am*x1[i-1] - ap*x0[i+1] + b[i])/ai
                else:
                    am, ai = A[i, i-1], A[i, i]
                    x_gs = (- am*x1[i-1] + b[i])/ai
            x1[i] = min(max((1-omega)*x0[i] + omega*x_gs, lower), upper)
            if np.linalg.norm(x1 - x0) <= epsilon:
                converged = True
                x0 = x1
        Vn = x1
        
        return Vn

    def prob_bs(Z_Grid, T, alpha, theta):
        # initialize
        row, col = Z_Grid.shape
        lower, upper = np.log(1 - alpha), np.log(1 + theta)
        BS_region = pd.DataFrame(columns=['p_s', 'p_b'])
        dp = 1/(row - 1)
        dt = 1/(col - 1)
        # assign
        for i in range(col):
            ps = (Z_Grid[:,i] == lower).sum()*dp
            pb = 1 - (Z_Grid[:,i] == upper).sum()*dp
            BS_region.loc[i*dt] = [ps, pb]
        
        return BS_region 

    def fully_implicit_fdm(lambd, u, sigma, rho, alpha, theta, T, I, N, epsilon, omega):

        dt, dp = T/N, 1/I
        lambda1, lambda2 = lambd[0], lambd[1]
        u1, u2 = u[0], u[1]
        upper, lower = np.log(1 + theta), np.log(1 - alpha)

        # initialization:
        Z_Grid = np.zeros([I+1, N+1])
        # terminal condition:
        Z_Grid[:, N] = np.log(1 - alpha)
        # boundary condition
        Z_Grid[0, :] = np.log(1 - alpha)
        Z_Grid[I, :] = np.log(1 + theta)

        # matrix A[(I-1)(I-1)]
        A = np.zeros([I-1, I-1])
        f_p = np.zeros(I-1)
        #F = np.zeros(I-1)
        for i in range(1, I, 1):
            b1 = -(lambda1 + lambda2)*dp*i + lambda2
            eta = 0.5*np.power(((u1-u2)*i*dp*(1 - i*dp)/sigma), 2)
            f_p[i-1] = (u1 - u2)*i*dp + u2 - np.power(sigma, 2)/2 - rho
            # upwind treatment
            if b1 >= 0:
                if i == 1:
                    f_d = -eta/np.power(dp, 2)*dt
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) + b1*dt/dp)
                    A[i-1, i] = -(eta*dt/np.power(dp, 2) + b1*dt/dp)
                elif i != I-1:
                    A[i-1, i-2] = -eta/np.power(dp, 2)*dt
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) + b1*dt/dp)
                    A[i-1, i] = -(eta*dt/np.power(dp, 2) + b1*dt/dp)
                else:
                    A[i-1, i-2] = -eta*dt/np.power(dp, 2)
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) + b1*dt/dp)
                    f_u = -(eta*dt/np.power(dp, 2) + b1*dt/dp) 
            else:
                if i == 1:
                    f_d = -(eta*dt/np.power(dp, 2) - b1*dt/dp)
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) - b1*dt/dp)
                    A[i-1, i] = -eta*dt/np.power(dp, 2)
                elif i != I-1:
                    A[i-1, i-2] = -(eta*dt/np.power(dp, 2) - b1*dt/dp)
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) - b1*dt/dp)
                    A[i-1, i] = -eta*dt/np.power(dp, 2)
                else:
                    A[i-1, i-2] = -(eta*dt/np.power(dp, 2) - b1*dt/dp)
                    A[i-1, i-1] = (1 + 2*eta*dt/np.power(dp, 2) - b1*dt/dp)
                    f_u = -eta*dt/np.power(dp, 2)

        # for loop
        for n in range(N-1, -1, -1):
            V = Z_Grid[1:I, n+1]
            size = len(V)
            b = V + f_p*dt
            b[0] = b[0] + f_d*lower 
            b[-1] = b[-1] + f_u*upper
            Vn = projectSOR(A, b, V, size, epsilon, omega, upper, lower)
            Z_Grid[1:I, n] = Vn.reshape([I-1])

        return Z_Grid
    
    
    pass