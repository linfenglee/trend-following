import numpy as np
import pandas as pd
from pandas import DataFrame
import plotly
import plotly.graph_objs as go

from trend_following_objects import BoundaryData


class BacktestEngine(object):
    """"""

    def __init__(
            self,
            bull_probability: DataFrame,
            bs_boundary: BoundaryData
    ):
        """"""

        self.pts = bull_probability
        self.vt_symbol = bs_boundary.vt_symbol
        self.buy_boundary = bs_boundary.buy_boundary
        self.sell_boundary = bs_boundary.sell_boundary

    def __str__(self):
        """"""

        return f"Back-test Engine for {self.vt_symbol} Trend Following Strategy"

    def __del__(self):
        """"""

        pass

    def get_position(self, initial_pos=0) -> DataFrame:
        """"""

        pos = initial_pos
        positions = DataFrame(columns=["price", "return", "position"])
        for trade_dt in self.pts.index:
            pt = self.pts.loc[trade_dt, "pt"]
            price = self.pts.loc[trade_dt, "price"]
            ret = self.pts.loc[trade_dt, "return"]
            if pt > self.buy_boundary:
                pos = 1
            elif pt < self.sell_boundary:
                pos = 0

            positions.loc[trade_dt] = [price, ret, pos]

        return positions

    def backtest_main(self, initial_pos=0):
        """"""

        positions = self.get_position(initial_pos)

        positions["strategy_return"] = positions["position"] * positions["return"]
        positions["wealth"] = np.cumprod(positions["strategy_return"] + 1)
        positions["holding"] = np.cumprod(positions["return"] + 1)

        trace_wealth = go.Scatter(
            x=positions.index,
            y=positions["wealth"],
            mode="lines",
            name="Trend Following Strategy"
        )

        trace_holding = go.Scatter(
            x=positions.index,
            y=positions["holding"],
            mode="lines",
            name="Buy & Hold Strategy"
        )

        data = [trace_wealth, trace_holding]
        layout = go.Layout(
            legend={"x": 1, "y": 1},
            title=f"Strategy Performance - {self.vt_symbol}"
        )
        fig = go.Figure(data=data, layout=layout)

        plotly.offline.plot(fig, filename='backtesting_result.html')


if __name__ == "__main__":
    """"""

    pass

