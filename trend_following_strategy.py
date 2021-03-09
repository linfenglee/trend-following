from typing import Tuple, List
import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly
import plotly.graph_objs as go

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
        """"""
        pass

    def get_regimes(self) -> List:
        """"""

        if not self.regimes:
            self.find_bull_bear()
        return self.regimes

    def get_initial_state(self) -> Tuple:
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

    def find_bull_bear(self) -> None:
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

    def separate_bull_bear(self) -> None:
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

    def parameter_estimation(self) -> None:
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

    def get_para_info(self) -> ParameterData:
        """
        Get Estimated Parameters Info
        """

        # divide into bull and bear market regimes
        self.find_bull_bear()
        self.separate_bull_bear()

        # parameter estimations
        self.parameter_estimation()

        return self.para_info

    def get_plot_info(self) -> Tuple:
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

    def plot_bull_bear(self) -> None:
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

        # plotly.offline.init_notebook_mode()
        # plotly.offline.iplot(fig, filename='scatter-mode')
        plotly.offline.plot(fig, filename='bull_bear_regime.html')


class TrendFollowing(object):

    def __init__(
            self,
            market_info: MarketInfo,
            model_info: ModelData,
            para_info: ParameterData
    ):
        """"""

        # Market Info
        self.rho = market_info.rho
        self.alpha = market_info.alpha
        self.theta = market_info.theta
        self.upper_boundary = market_info.upper_boundary
        self.lower_boundary = market_info.lower_boundary

        # Model Info
        self.T = model_info.T
        self.I = model_info.I
        self.N = model_info.N
        self.dt = model_info.dt
        self.dp = model_info.dp
        self.epsilon = model_info.epsilon
        self.omega = model_info.omega

        # Parameter Info
        self.bull_mu = para_info.bull_mu
        self.bear_mu = para_info.bear_mu
        self.bull_sigma = para_info.bull_sigma
        self.bear_sigma = para_info.bear_sigma
        self.const_sigma = para_info.constant_sigma
        self.bull_lambda = para_info.bull_lambda
        self.bear_lambda = para_info.bear_lambda

        self.matrix_A = np.zeros([self.I-1, self.I-1])

    def reset_matrix(self) -> None:
        """"""
        # matrix A[(I-1)(I-1)]
        self.matrix_A = np.zeros([self.I-1, self.I-1])

    def get_calc_sigma(self, market_regime: str) -> float:
        """"""

        if market_regime == "bull":
            sigma = self.bull_sigma
        elif market_regime == "bear":
            sigma = self.bear_sigma
        else:
            sigma = self.const_sigma

        return sigma

    def init_grid_z(self) -> np.array:
        """"""

        # initialization:
        grid_z = np.zeros([self.I + 1, self.N + 1])

        # terminal condition:
        grid_z[:, self.N] = np.log(1 - self.alpha)

        # boundary condition
        grid_z[0, :] = np.log(1 - self.alpha)
        grid_z[self.I, :] = np.log(1 + self.theta)

        return grid_z

    def projected_sor(self, b, vector_v, n):
        """
        Implement Projected SOR method to find buy & sell boundary.
        """

        converged = False
        x0 = vector_v
        x1 = np.zeros(n)
        while not converged:
            for i in range(n):
                if i == 0:
                    ai, ap = self.matrix_A[i, i], self.matrix_A[i, i+1]
                    x_gs = (-ap*x0[i+1] + b[i])/ai
                elif i != (n-1):
                    am, ai, ap = self.matrix_A[i, i-1], self.matrix_A[i, i], self.matrix_A[i, i+1]
                    x_gs = (- am*x1[i-1] - ap*x0[i+1] + b[i])/ai
                else:
                    am, ai = self.matrix_A[i, i-1], self.matrix_A[i, i]
                    x_gs = (- am*x1[i-1] + b[i])/ai
                x1[i] = min(
                    max((1-self.omega)*x0[i] + self.omega*x_gs, self.lower_boundary),
                    self.upper_boundary
                )
            if np.linalg.norm(x1 - x0) <= self.epsilon:
                converged = True
            x0 = x1
        vector_vn = x1
        
        return vector_vn

    def prob_bs(self, z_array: np.array) -> np.array:
        """
        Calculate the bug & sell probability
        """

        # initialize
        row, col = z_array.shape
        bs_region = pd.DataFrame(columns=['p_s', 'p_b'])

        # assign
        for i in range(col):
            ps = (z_array[:, i] == self.lower_boundary).sum() * self.dp
            pb = 1 - (z_array[:, i] == self.upper_boundary).sum() * self.dp
            bs_region.loc[i * self.dt] = [ps, pb]
        
        return bs_region

    def fully_implicit_fdm(self, market_regime: str) -> np.array:
        """"""

        # get sigma for calculation
        sigma = self.get_calc_sigma(market_regime)

        # initialize A matrix
        self.reset_matrix()

        # initialize Z Grid
        grid_z = self.init_grid_z()

        # F = np.zeros(I-1)
        f_p = np.zeros(self.I-1)

        for i in range(1, self.I, 1):

            b1 = -(self.bull_lambda + self.bear_lambda)*self.dp*i + self.bear_lambda
            eta = 0.5*np.power(
                ((self.bull_mu-self.bear_mu)*i*self.dp*(1 - i*self.dp)/sigma), 2
            )
            f_p[i-1] = (self.bull_mu - self.bear_mu)*i*self.dp + self.bear_mu - np.power(sigma, 2)/2 - self.rho

            para_a = eta * self.dt / np.power(self.dp, 2)
            para_b = b1 * self.dt / self.dp

            # upwind treatment
            if b1 >= 0:
                if i == 1:
                    f_d = -para_a
                    self.matrix_A[i-1, i-1] = 1 + 2 * para_a + para_b
                    self.matrix_A[i-1, i] = -(para_a + para_b)
                elif i != self.I-1:
                    self.matrix_A[i-1, i-2] = -para_a
                    self.matrix_A[i-1, i-1] = 1 + 2 * para_a + para_b
                    self.matrix_A[i-1, i] = -(para_a + para_b)
                else:
                    self.matrix_A[i-1, i-2] = -para_a
                    self.matrix_A[i-1, i-1] = (1 + 2 * para_a + para_b)
                    f_u = -(para_a + para_b)
            else:
                if i == 1:
                    f_d = -(para_a - para_b)
                    self.matrix_A[i-1, i-1] = (1 + 2 * para_a - para_b)
                    self.matrix_A[i-1, i] = -para_a
                elif i != self.I-1:
                    self.matrix_A[i-1, i-2] = -(para_a - para_b)
                    self.matrix_A[i-1, i-1] = (1 + 2 * para_a - para_b)
                    self.matrix_A[i-1, i] = -para_a
                else:
                    self.matrix_A[i-1, i-2] = -(para_a - para_b)
                    self.matrix_A[i-1, i-1] = (1 + 2 * para_a - para_b)
                    f_u = -para_a

        # for loop
        for n in range(self.N-1, -1, -1):
            vector_v = grid_z[1:self.I, n+1]
            size = len(vector_v)
            b = vector_v + f_p * self.dt
            b[0] = b[0] + f_d * self.lower_boundary
            b[-1] = b[-1] + f_u * self.upper_boundary
            vector_vn = self.projected_sor(b, vector_v, size)
            grid_z[1:self.I, n] = vector_vn.reshape([self.I-1])

            percentage = int((self.N - n) / self.N * 100)
            output = "#" * percentage + " " + f"{percentage}%"
            print(output, end="\r")

        return grid_z

    @staticmethod
    def bs_plot(bs_data: np.array) -> None:
        """"""

        trace1 = go.Scatter(
            x=bs_data.index,
            y=bs_data["p_s"],
            mode="lines",
            name="prob sell"
        )
        trace2 = go.Scatter(
            x=bs_data.index,
            y=bs_data["p_b"],
            mode="lines",
            name="prob buy"
        )

        data = [trace1, trace2]
        layout = go.Layout(
            legend={"x": 1, "y": 1},
            title="Buy Region & Sell Region"
        )
        fig = go.Figure(data=data, layout=layout)

        plotly.offline.plot(fig, filename='bs_region.html')

    def main(self):
        """"""

        # # get bull
        # bull_z_grid = self.fully_implicit_fdm("bull")
        # bull_bs = self.prob_bs(bull_z_grid)

        # # get bear
        # bear_z_grid = self.fully_implicit_fdm("bear")
        # bear_bs = self.prob_bs(bear_z_grid)

        # get constant
        const_z_grid = self.fully_implicit_fdm("const")
        const_bs = self.prob_bs(const_z_grid)

        # plot sell & buy region
        self.bs_plot(const_bs)

        return const_bs, const_z_grid


if __name__ == '__main__':
    """"""

    test_market_info = MarketInfo()
    test_model_info = ModelData()
    test_para_info = ParameterData()

    trend_following = TrendFollowing(
        test_market_info, test_model_info, test_para_info
    )

    constant_bs, z_grid = trend_following.main()

