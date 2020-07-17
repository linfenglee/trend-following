"""
Basic data structure used for trend following strategy.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np



@dataclass
class BaseData:
    """
    Any data object needs a vt_symbol as source
    and should inherit base data.
    """

    vt_symbol: str


@dataclass
class MarketInfo(BaseData):
    """
    Market Information
    """

    rho: float = 0       # interest rate
    alpha: float = 0     # buy transaction fee
    theta: float = 0      # sell transaction fee

    def __post_init__(self):
        """"""
        self.upper_boundary = np.log(1 + self.theta)
        self.lower_boundary = np.log(1 - self.alpha)
        

@dataclass
class ModelData(BaseData):
    """
    Basic setting to implement fully-implicit finite difference method
    """

    T: float
    I: int
    N: int

    def __post_init__(self):
        """"""
        self.dt = self.T / self.N
        self.dp = 1 / self.I

    epsilon: float
    omega: float


@dataclass
class ParameterData(BaseData):
    """
    Parameter Data used in trend following strategy
    """
    bull_mu: float = 0
    bear_mu: float = 0

    bull_sigma: float = 0
    bear_sigma: float = 0
    constant_sigma: float = 0

    bull_lambda: float = 0
    bear_lambda: float = 0




    