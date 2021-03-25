from datetime import datetime
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go

from trend_following_objects import (
    MarketInfo, ModelData, ParameterData, RegimeInfo
)
from trend_following_engine import TrendFollowingEngine
from trend_following_estimate import EstimationEngine


def test_main():
    """"""

    test_market_info = MarketInfo()
    test_model_info = ModelData()
    test_para_info = ParameterData()

    trend_following = TrendFollowingEngine(
        test_market_info, test_model_info, test_para_info
    )

    constant_bs = trend_following.main()

    return constant_bs


def main(
        source: str,
        ts_code: str,
        price_type: str,
        start_time: datetime,
        start_prob: float,
        bull_profit: float,
        bear_loss: float
):
    """"""

    regime_data = RegimeInfo(ts_code, bull_profit, bear_loss)

    est_engine = EstimationEngine(source, ts_code, price_type)
    est_engine.set_regime_info(regime_data)
    para_data = est_engine.estimate_para()
    est_engine.set_para_info(para_data)
    pts = est_engine.estimate_prob(start_time, start_prob)
    prob = pts["pt"].to_numpy()[-1]

    est_engine.est_para.plot_bull_bear()
    est_engine.est_prob.pts_plot(pts)

    market_info = MarketInfo(vt_symbol=ts_code)
    model_info = ModelData(vt_symbol=ts_code)

    trend_following = TrendFollowingEngine(
        market_info, model_info, para_data
    )

    constant_bs = trend_following.main()

    buy_boundary = constant_bs["p_b"].to_numpy()[0]
    sell_boundary = constant_bs["p_s"].to_numpy()[0]

    if prob >= buy_boundary:
        print(f"Long {ts_code} Tomorrow")
    elif prob <= sell_boundary:
        print(f"Sell {ts_code} Tomorrow")
    else:
        print(f"Hold or Close {ts_code} Tomorrow")

    return pts, constant_bs


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
    prob_t, const_bs = main(
        data_source,
        ts_code,
        price_type,
        start_time,
        start_prob,
        bull_profit,
        bear_loss
    )
