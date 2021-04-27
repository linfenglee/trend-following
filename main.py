from datetime import datetime
from typing import Tuple
from pandas import DataFrame

import plotly
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from trend_following_objects import (
    MarketInfo, ModelData, ParameterData, RegimeInfo, BoundaryData
)
from trend_following_engine import TrendFollowingEngine
from trend_following_estimate import EstimationEngine, EstimationParameter
from trend_following_backtesting import BacktestEngine


def test_main() -> BoundaryData:
    """
    Test Main for pre-set parameters
    """

    test_market_info = MarketInfo()
    test_model_info = ModelData()
    test_para_info = ParameterData()

    trend_following_engine = TrendFollowingEngine(
        test_market_info, test_model_info, test_para_info
    )

    bs_boundary = trend_following_engine.main()

    return bs_boundary


def main_plot(
        pts: DataFrame,
        bs_boundary: BoundaryData,
        estimate_parameter: EstimationParameter
) -> None:
    """"""

    fig = make_subplots(rows=2, cols=1)

    trace_pt = go.Scatter(
        x=pts.index,
        y=pts["pt"],
        mode="lines",
        name="pt"
    )

    trace_sell = go.Scatter(
        x=pts.index,
        y=[bs_boundary.sell_boundary] * len(pts.index),
        mode="lines",
        name="sell boundary"
    )

    trace_buy = go.Scatter(
        x=pts.index,
        y=[bs_boundary.buy_boundary] * len(pts.index),
        mode="lines",
        name="buy boundary"
    )

    trace_stock, trace_point = estimate_parameter.initialize_plot_trace()

    fig.add_traces([trace_pt, trace_sell, trace_buy], rows=2, cols=1)
    fig.add_traces([trace_stock, trace_point], rows=1, cols=1)

    plotly.offline.plot(fig, filename='main_plot.html')


def main(
        source: str,
        ts_code: str,
        price_type: str,
        start_time: datetime,
        start_prob: float,
        bull_profit: float,
        bear_loss: float
) -> Tuple:
    """"""

    regime_data = RegimeInfo(ts_code, bull_profit, bear_loss)

    est_engine = EstimationEngine(source, ts_code, price_type)
    est_engine.set_regime_info(regime_data)
    para_data = est_engine.estimate_para()
    est_engine.set_para_info(para_data)
    pts = est_engine.estimate_prob(start_time, start_prob)
    prob = pts["pt"].to_numpy()[-1]

    # est_engine.est_para.plot_bull_bear()
    # est_engine.est_prob.pts_plot(pts)

    market_info = MarketInfo(vt_symbol=ts_code)
    model_info = ModelData(vt_symbol=ts_code)

    trend_following = TrendFollowingEngine(
        market_info, model_info, para_data
    )

    bs_boundary = trend_following.main()

    main_plot(pts, bs_boundary, est_engine.est_para)

    if prob >= bs_boundary.buy_boundary:
        print(f"Long {ts_code} Tomorrow")
    elif prob <= bs_boundary.sell_boundary:
        print(f"Sell {ts_code} Tomorrow")
    else:
        print(f"Hold or Close {ts_code} Tomorrow")

    backtesting_engine = BacktestEngine(pts, bs_boundary)
    backtesting_engine.backtest_main(initial_pos=0)

    return pts, bs_boundary


if __name__ == '__main__':
    """"""

    data_source = "yahoo"
    ts_code = "399001.SZ"
    price_type = "close"
    start_time = datetime(2011, 10, 1)
    start_prob = 0.5
    bull_profit = 0.3
    bear_loss = 0.3

    # const_bs = test_main()
    prob_t, boundary_data = main(
        data_source,
        ts_code,
        price_type,
        start_time,
        start_prob,
        bull_profit,
        bear_loss
    )
