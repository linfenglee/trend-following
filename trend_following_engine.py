from typing import Tuple
import time
from copy import deepcopy
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import diags
from scipy.sparse.linalg import inv
from scipy.linalg import solve_banded
from scipy.sparse.linalg import spsolve
import plotly
import plotly.graph_objs as go

from trend_following_objects import (
    State,
    MarketInfo,
    ModelData,
    ParameterData,
    BoundaryData
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

        # vt_symbol
        self.vt_symbol = market_info.vt_symbol

        # Market Info
        self.rho = market_info.rho
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

    def calc_p0(self, market_regime: State) -> float:
        """
        ps -> p0 when t -> T
        """

        sigma = self.get_calc_sigma(market_regime)
        denominator = self.bull_mu - self.bear_mu
        nominator = self.rho - self.bear_mu + np.power(sigma, 2) / 2
        p0 = nominator / denominator

        return p0

    def calc_a(self) -> float:
        """
        when t > a / (mu1 - rho - sigma^2/2) such that pb(t) = 1
        """

        a = np.log(self.upper_boundary / self.lower_boundary)
        return a

    def calc_delta_a(self, market_regime: State) -> float:
        """
        when t > a / (mu1 - rho - sigma^2/2) such that pb(t) = 1
        """

        a = self.calc_a()
        sigma = self.get_calc_sigma(market_regime)
        delta = a / (self.bull_mu - self.rho - np.power(sigma, 2) / 2)

        return delta

    def reset_matrix(self) -> None:
        """"""

        # matrix A[(I-1)(I-1)]
        self.matrix_A = np.zeros([self.I-1, self.I-1])

    def get_calc_sigma(self, market_regime: State) -> float:
        """"""

        if market_regime == State.Bull:
            sigma = self.bull_sigma
        elif market_regime == State.Bear:
            sigma = self.bear_sigma
        else:
            sigma = self.const_sigma

        return sigma

    # def init_grid_z(self) -> np.array:
    #     """"""
    #
    #     # initialization:
    #     grid_z = np.zeros([self.I + 1, self.N + 1])
    #
    #     # terminal condition:
    #     grid_z[:, self.N] = self.lower_boundary
    #
    #     # boundary condition
    #     grid_z[0, :] = self.lower_boundary
    #     grid_z[self.I, :] = self.upper_boundary
    #
    #     return grid_z

    def prob_bs(self, vn: np.array) -> Tuple:
        """
        Calculate the bug & sell probability
        """

        # initialize
        add = (self.I + 1 - len(vn)) / 2

        ps = ((vn <= self.lower_boundary).sum() + add) * self.dp
        pb = 1 - ((vn >= self.upper_boundary).sum() + add) * self.dp

        return ps, pb

    def show_progress(self, n: int) -> None:
        """
        Show the progress
        """

        percentage = int((self.N - n) / self.N * 100)
        output = "#" * percentage + " " + f"{percentage}%"
        print(output, end="\r")

        return

    def projected_sor(self, b: np.array, vector_v: np.array) -> np.array:
        """
        Implement Projected SOR method to find buy & sell boundary.
        """

        x0 = vector_v
        n = len(vector_v)
        x1 = np.zeros(n)
        converged = False
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

    def fully_implicit_fdm(self, market_regime: State) -> DataFrame:
        """
        Fully Implicit Finite Difference Method
        """

        # get sigma for calculation
        sigma = self.get_calc_sigma(market_regime)

        # initialize A matrix
        self.reset_matrix()

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
        bs_region = pd.DataFrame(columns=['p_s', 'p_b'])
        bs_region.loc[self.N] = [1, 1]
        vector_v = np.array([self.lower_boundary for _ in range(self.I-1)])
        for n in range(self.N-1, -1, -1):

            # adjust vector b
            b = vector_v + f_p * self.dt
            b[0] = b[0] + f_d * self.lower_boundary
            b[-1] = b[-1] + f_u * self.upper_boundary

            vector_vn = self.projected_sor(b, vector_v)
            bs_region.loc[n] = self.prob_bs(vector_vn)

            vector_v = vector_vn

            self.show_progress(n)

        bs_region.index = bs_region.index / self.N

        return bs_region

    def penalty_method(self, market_regime: State) -> DataFrame:
        """
        Penalty Method
        """

        # get sigma for calculation
        sigma = self.get_calc_sigma(market_regime)

        p = np.array([self.dp * i for i in range(self.I + 1)])

        # upwind treatment
        eta = 0.5 * np.power(
            (self.bull_mu - self.bear_mu) * p * (1 - p) / sigma, 2
        ) / np.power(self.dp, 2)
        b1 = (-(self.bull_lambda + self.bear_lambda) * p + self.bear_lambda) / self.dp
        f_p = (self.bull_mu - self.bear_mu) * p + (self.bear_mu - self.rho - 0.5*np.power(sigma, 2))

        # initialize and adjust sparse diag values
        left = -eta + b1 * (b1 < 0)
        middle = 1 / self.dt + 2 * eta + np.abs(b1)
        right = -eta - b1 * (b1 > 0)
        left[-1], right[0] = 0, 0

        # get initial vn+1 from n+1 values
        vn = np.array([self.lower_boundary for _ in range(self.I+1)])
        vn1 = np.array([self.lower_boundary for _ in range(self.I+1)])

        # for loop
        bs_region = pd.DataFrame(columns=['p_s', 'p_b'])
        bs_region.loc[self.N] = [1, 1]
        for n in range(self.N - 1, -1, -1):

            while True:

                # construct indicator function
                ind_upper = self.beta * (vn > self.upper_boundary)
                ind_lower = self.beta * (vn < self.lower_boundary)

                b = vn1 / self.dt + self.upper_boundary * ind_upper + self.lower_boundary * ind_lower + f_p
                middle_first = middle[0] + ind_lower[0] + ind_lower[0]
                middle_last = middle[-1] + ind_lower[-1] + ind_upper[-1]
                self.matrix_A = diags(
                    [left[1:], middle+ind_upper+ind_lower, right[:-1]],
                    offsets=[-1, 0, 1], shape=(self.I+1, self.I+1), format="csc"
                )

                # adjustment for boundary condition
                b[0] = middle_first * self.lower_boundary
                b[-1] = middle_last * self.upper_boundary

                vn_new = spsolve(self.matrix_A, b)
                iter_error = np.linalg.norm(vn_new - vn) / np.linalg.norm(vn)

                if iter_error <= self.epsilon:
                    vn = vn_new
                    break
                else:
                    vn = vn_new

            bs_region.loc[n] = self.prob_bs(vn_new)
            vn1 = vn_new

            self.show_progress(n)

        bs_region.index = bs_region.index / self.N

        return bs_region

    def get_boundaries(self, bs_data: DataFrame) -> BoundaryData:
        """"""

        sell_boundary = bs_data["p_s"].to_numpy()[-1]
        buy_boundary = bs_data["p_b"].to_numpy()[-1]
        boundary_data = BoundaryData(
            vt_symbol=self.vt_symbol, sell_boundary=sell_boundary, buy_boundary=buy_boundary
        )
        # print(f"Sell Boundary: {sell_boundary} | Buy Boundary: {buy_boundary}")

        return boundary_data

    @staticmethod
    def bs_plot(bs_data: DataFrame) -> None:
        """
        Plot Buy Boundary & Sell Boundary
        """

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

    def main(self) -> BoundaryData:
        """
        Main Function to Implement Trend Following Engine
        """

        # # get bull
        # bull_z_grid = self.fully_implicit_fdm("bull")
        # bull_bs = self.prob_bs(bull_z_grid)

        # # get bear
        # bear_z_grid = self.fully_implicit_fdm("bear")
        # bear_bs = self.prob_bs(bear_z_grid)

        # # get constant via fully implicit FDM
        # const_bs = self.fully_implicit_fdm("const")

        # get constant via penalty method FDM
        const_bs = self.penalty_method(State.Constant)

        # plot sell & buy region
        self.bs_plot(const_bs)

        # get buy & sell boundary
        boundary_data = self.get_boundaries(const_bs)

        return boundary_data


if __name__ == '__main__':
    """
    An example for trend following engine
    """

    test_market_info = MarketInfo()
    test_model_info = ModelData()
    test_para_info = ParameterData()

    trend_following_engine = TrendFollowingEngine(
        test_market_info, test_model_info, test_para_info
    )

    bs_boundaries = trend_following_engine.main()

