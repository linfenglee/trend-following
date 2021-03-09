from importlib import reload
import numpy as np
import pandas as pd

import plotly
import plotly.graph_objs as go


from trend_following_objects import (
    MarketInfo, ModelData, ParameterData, RegimeInfo
)

import trend_following_engine
reload(trend_following_engine)
from trend_following_engine import TrendFollowingEngine


def para_estimate():
    pass


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


if __name__ == '__main__':
    """"""

    const_bs = test_main()
