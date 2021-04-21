from typing import Tuple, List
from copy import deepcopy
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import diags
import plotly
import plotly.graph_objs as go

from trend_following_objects import (
    MarketInfo,
    RegimeInfo,
    ModelData,
    ParameterData
)


class TrendFollowingEngine(object):
    """
    Trend Following Engine
    """

    def __init__(
            self,
            market_info: MarketInfo,
            model_info: ModelData,
            para_info: ParameterData
    ):
        """"""

        # Market Info
        self.rho = market_info.rho
        # self.alpha = market_info.alpha
        # self.theta = market_info.theta
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
        self.beta = model_info.beta

        # Parameter Info
        self.bull_mu = para_info.bull_mu
        self.bear_mu = para_info.bear_mu
        self.bull_sigma = para_info.bull_sigma
        self.bear_sigma = para_info.bear_sigma
        self.const_sigma = para_info.constant_sigma
        self.bull_lambda = para_info.bull_lambda
        self.bear_lambda = para_info.bear_lambda

        self.matrix_A = np.zeros([self.I-1, self.I-1])

    def __str__(self):
        """"""
        return f"Trend Following Engine"

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
        grid_z[:, self.N] = self.lower_boundary

        # boundary condition
        grid_z[0, :] = self.lower_boundary
        grid_z[self.I, :] = self.upper_boundary

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
            ps = (z_array[:, i] <= self.lower_boundary).sum() * self.dp
            pb = 1 - (z_array[:, i] >= self.upper_boundary).sum() * self.dp
            bs_region.loc[i * self.dt] = [ps, pb]
        
        return bs_region

    def fully_implicit_fdm(self, market_regime: str) -> np.array:
        """
        Fully Implicit FDM
        """

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

    def penalty_method(self, market_regime: str) -> np.array:
        """
        Penalty Method
        """

        # get sigma for calculation
        sigma = self.get_calc_sigma(market_regime)

        # initialize Z Grid
        grid_z = self.init_grid_z()

        p = np.array([
            self.dp * i for i in range(self.I + 1)
        ])

        # upwind treatment
        eta = 0.5 * np.power(
            (self.bull_mu - self.bear_mu) * p * (1 - p) / sigma, 2
        ) / self.dp / self.dp
        b1 = -((self.bull_lambda - self.bear_lambda) * p + self.bear_lambda) / self.dp
        f_p = (self.bull_mu - self.bear_mu) * p + (self.bear_mu - self.rho - 0.5*sigma**2)

        left = -eta + b1 * (b1 < 0)
        middle = 1 / self.dt + 2 * eta + np.abs(b1)
        right = -eta - b1 * (b1 > 0)

        # for loop
        for n in range(self.N - 1, -1, -1):

            # get initial vn from n+1 values
            vn1 = deepcopy(grid_z[:, n + 1])
            vn = deepcopy(grid_z[:, n + 1])

            while True:

                # construct indicator function
                ind_upper = self.beta * (vn > self.upper_boundary)
                ind_lower = self.beta * (vn < self.lower_boundary)

                b = vn1 / self.dt + self.upper_boundary * ind_upper + self.lower_boundary * ind_lower + f_p

                self.matrix_A = diags(
                    [left[1:], middle+ind_upper+ind_lower, right[:-1]], offsets=[-1, 0, 1]
                ).toarray()

                # adjustment for boundary condition
                self.matrix_A[0, 1] = 0
                self.matrix_A[-1, -2] = 0
                b[0] = self.matrix_A[0, 0] * self.lower_boundary
                b[-1] = self.matrix_A[-1, -1] * self.upper_boundary

                vn_new = np.linalg.solve(self.matrix_A, b)

                if np.linalg.norm(vn_new - vn) <= self.epsilon:
                    break
                else:
                    vn = vn_new

            grid_z[:, n] = vn_new

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

    def main(self) -> DataFrame:
        """"""

        # # get bull
        # bull_z_grid = self.fully_implicit_fdm("bull")
        # bull_bs = self.prob_bs(bull_z_grid)

        # # get bear
        # bear_z_grid = self.fully_implicit_fdm("bear")
        # bear_bs = self.prob_bs(bear_z_grid)

        # # get constant via fully implicit FDM
        # const_z_grid = self.fully_implicit_fdm("const")
        # const_bs = self.prob_bs(const_z_grid)

        # get constant via penalty method FDM
        const_z_grid = self.penalty_method("const")
        const_bs = self.prob_bs(const_z_grid)

        # plot sell & buy region
        self.bs_plot(const_bs)

        return const_bs


if __name__ == '__main__':
    """"""

    test_market_info = MarketInfo()
    test_model_info = ModelData()
    test_para_info = ParameterData()

    trend_following = TrendFollowingEngine(
        test_market_info, test_model_info, test_para_info
    )

    constant_bs = trend_following.main()

