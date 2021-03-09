from datetime import *
import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly
import plotly.graph_objs as go

import rqdatac as rq
from rqdatac import *

from trend_following_objects import (
    MarketInfo,
    RegimeInfo,
    ModelData,
    ParameterData
)


class EstimationParameter(object):
    """
    Estimate parameters, such as lambda, mu, sigma
    """

    def __init__(
            self, data: DataFrame, price_type: str, regime_info: RegimeInfo
    ):
        """
        Initialize the Estimation Parameters
        """

        self.data = data
        self.vt_symbol = regime_info.vt_symbol
        self.bull_p = regime_info.bull_profit
        self.bear_l = regime_info.bear_loss

        self.para_info = ParameterData(self.vt_symbol)

        self.regimes = []
        self.close_array = self.data[price_type]

        self.bull_data = None
        self.bull_periods = None
        self.bear_data = None
        self.bear_periods = None

    def __str__(self):
        """"""
        return f"Estimate Parameters for {self.vt_symbol}"

    def __del__(self):
        """"""
        print(f"Delete {self.vt_symbol} Parameter Estimation Object")

    def close(self):
        pass

    def get_regimes(self):
        """"""
        if not self.regimes:
            self.find_bull_bear()
        return self.regimes

    def get_initial_state(self):
        """"""
        initial_state = None
        start_price = self.close_array[0]
        bull_price = start_price * (1 + self.bull_p)
        bear_price = start_price * (1 - self.bear_l)
        for price in self.close_array:
            if price > bull_price:
                initial_state = "bull"
                break
            elif price < bear_price:
                initial_state = "bear"
                break
        if initial_state == "bull":
            bull_date = self.close_array[self.close_array > bull_price].index[0]
            bull_point = self.close_array[self.close_array.index < bull_date].min()
            start_date = self.close_array[self.close_array == bull_point].index[0]
        elif initial_state == "bear":
            bear_date = self.close_array[self.close_array < bear_price].index[0]
            bear_point = self.close_array[self.close_array.index < bear_date].max()
            start_date = self.close_array[self.close_array == bear_point].index[0]
        else:
            print(f"{self.vt_symbol} data depict no bull or bear regime")
            start_date = self.close_array.index[0]

        return initial_state, start_date

    def find_bull_bear(self):
        """
        Find the bull and bear market regimes,
        and keep the info in self.regimes (Dict)
        """

        signal, start_date = self.get_initial_state()
        if signal is None:
            return

        close_array = self.close_array[self.close_array.index >= start_date]
        max_s, min_s = close_array[0], close_array[0]
        self.regimes.clear()
        for idx in close_array.index:
            if signal == "bull":
                if close_array[idx]/max_s < 1 - self.bear_l:
                    closes = close_array[close_array.index <= idx]
                    start = closes[closes == min_s].index[-1]
                    end = closes[closes == max_s].index[-1]
                    if start > end:
                        start = closes[closes == min_s].index[-2]
                    bull_info = {
                        "state": signal,
                        "start_date": start,
                        "end_date": end,
                        "min_point": min_s,
                        "max_point": max_s
                    }
                    self.regimes.append(bull_info)
                    signal = "bear"
                    min_s = close_array[idx]
                else:
                    max_s = max(close_array[idx], max_s)
            elif signal == "bear":
                if self.close_array[idx] / min_s > 1 + self.bull_p:
                    closes = close_array[close_array.index <= idx]
                    start = closes[closes == max_s].index[-1]
                    end = closes[closes == min_s].index[-1]
                    if start > end:
                        start = closes[closes == min_s].index[-2]
                    bear_info = {
                        "state": signal,
                        "start_date": start,
                        "end_date": end,
                        "min_point": min_s,
                        "max_point": max_s
                    }
                    self.regimes.append(bear_info)
                    signal = "bull"
                    max_s = close_array[idx]
                else:
                    min_s = min(close_array[idx], min_s)

    def separate_bull_bear(self):
        """
        Separate bull and bear for further estimation
        """

        bull_periods, bear_periods = [], []
        bull_data, bear_data = [], []
        for regime in self.regimes:
            start_date = pd.to_datetime(regime["start_date"])
            end_date = pd.to_datetime(regime["end_date"])
            data_batch = self.data[
                (self.data.index > start_date) & (self.data.index < end_date)
            ]["log_rtn"].values.tolist()
            period = (pd.to_datetime(regime["end_date"]) - pd.to_datetime(regime["start_date"])).days / 365
            if regime["state"] == "bull":
                bull_periods.append(period)
                bull_data.extend(data_batch)
            elif regime["state"] == "bear":
                bear_periods.append(period)
                bear_data.extend(data_batch)
            else:
                print("Invalid Market Regime:", regime)
        
        self.bull_data, self.bear_data = np.array(bull_data), np.array(bear_data)
        self.bull_periods, self.bear_periods = np.array(bull_periods), np.array(bear_periods)

    def parameter_estimation(self):
        """
        Estimate lambda, mu, and sigma
        """

        # lambda estimation
        self.para_info.bull_lambda = (1 / self.bull_periods).mean()
        self.para_info.bear_lambda = (1 / self.bear_periods).mean()

        # sigma estimation
        bull_sigma = np.sqrt(240 * np.power(self.bull_data, 2).mean())
        bear_sigma = np.sqrt(240 * np.power(self.bear_data, 2).mean())
        self.para_info.bull_sigma = bull_sigma
        self.para_info.bear_sigma = bear_sigma

        # constant sigma estimation
        bull_bear_data = self.data["log_rtn"].values[1:]
        constant_sigma = np.sqrt(240 * np.power(bull_bear_data, 2).mean())
        self.para_info.constant_sigma = constant_sigma

        # mu estimation
        bull_mu = np.power(bull_sigma, 2) / 2 + 240 * self.bull_data.mean()
        bear_mu = np.power(bear_sigma, 2) / 2 + 240 * self.bear_data.mean()
        self.para_info.bull_mu = bull_mu
        self.para_info.bear_mu = bear_mu

    def get_para_info(self):
        """
        Get Estimated Parameters Info
        """

        # divide into bull and bear market regimes
        self.find_bull_bear()
        self.separate_bull_bear()

        # parameter estimations
        self.parameter_estimation()

        return self.para_info

    def get_plot_info(self):
        """
        Get Bull & Bear Points Information
        """

        s_x, s_y = [], []
        for regime in self.regimes:
            if regime["state"] == "bull":
                s_y.extend(
                    [regime["min_point"], regime["max_point"]]
                )
                s_x.extend([
                    pd.to_datetime(regime["start_date"]),
                    pd.to_datetime(regime["end_date"])
                ])
            else:
                s_y.extend(
                    [regime["max_point"], regime["min_point"]]
                )
                s_x.extend([
                    pd.to_datetime(regime["start_date"]),
                    pd.to_datetime(regime["end_date"])
                ])

        return s_x, s_y

    def plot_bull_bear(self):
        """
        Plot Bull Bear Points in Price Figure
        """

        # find bull bear regimes
        self.find_bull_bear()

        # get bull bear points info
        s_x, s_y = self.get_plot_info()

        trace1 = go.Scatter(
            x=self.data.index,
            y=self.data["close"],
            mode="lines",
            name=self.vt_symbol
        )

        trace2 = go.Scatter(
            x=s_x,
            y=s_y,
            mode="markers",
            name="point"
        )

        data = [trace1, trace2]

        layout = go.Layout(
            legend={"x": 1, "y": 1},
            title=f"{self.vt_symbol} Bull Bear Regimes"
        )

        fig = go.Figure(data=data, layout=layout)

        plotly.offline.init_notebook_mode()
        plotly.offline.iplot(fig, filename='scatter-mode')


class TrendFollowing(object):

    def __init__(
            self, market_info: MarketInfo, para_info: ParameterData
    ):
        pass

    def projected_sor(self, A, b, V, n, epsilon, omega, upper, lower):
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

    def prob_bs(self, Z_Grid, T, alpha, theta):
        # initialize
        row, col = Z_Grid.shape
        lower, upper = np.log(1 - alpha), np.log(1 + theta)
        BS_region = pd.DataFrame(columns=['p_s', 'p_b'])
        dp = 1/(row - 1)
        dt = 1/(col - 1)
        # assign
        for i in range(col):
            ps = (Z_Grid[:, i] == lower).sum()*dp
            pb = 1 - (Z_Grid[:, i] == upper).sum()*dp
            BS_region.loc[i*dt] = [ps, pb]
        
        return BS_region 

    def fully_implicit_fdm(self, lambd, u, sigma, rho, alpha, theta, T, I, N, epsilon, omega):

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
            Vn = self.projectSOR(A, b, V, size, epsilon, omega, upper, lower)
            Z_Grid[1:I, n] = Vn.reshape([I-1])

        return Z_Grid

    pass