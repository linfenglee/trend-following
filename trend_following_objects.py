from enum import Enum
from dataclasses import dataclass
import numpy as np


ANNUAL_DAYS = 240


class State(Enum):
    """"""

    Bull = "bull"
    Bear = "bear"
    Constant = "constant"


@dataclass
class BaseData:
    """
    Any data object needs a vt_symbol as source
    and should inherit base data.
    """

    vt_symbol: str = ""


@dataclass
class MarketInfo(BaseData):
    """
    Market Information
    """

    __rho: float = 0.0679       # interest rate
    __alpha: float = 0.001     # buy transaction fee
    __theta: float = 0.001      # sell transaction fee

    @property
    def rho(self):
        return self.__rho

    @property
    def alpha(self):
        return self.__alpha

    @property
    def theta(self):
        return self.__theta

    def __post_init__(self):
        """"""
        self.upper_boundary = np.log(1 + self.__theta)
        self.lower_boundary = np.log(1 - self.__alpha)
        

@dataclass
class RegimeInfo(BaseData):
    """
    Market Regime Information
    """

    bull_profit: float = 0.2
    bear_loss: float = 0.2


@dataclass
class ModelData(BaseData):
    """
    Basic setting to implement fully-implicit finite difference method
    """

    T: float = 1
    I: int = 2_000
    N: int = 1_000

    def __post_init__(self):
        """"""
        self.dt = self.T / self.N
        self.dp = 1 / self.I

    epsilon: float = 1.0e-6
    omega: float = 1.6   # relax parameter
    beta: float = 1.0e7  # penalty method


@dataclass
class ParameterData(BaseData):
    """
    Parameter Data used in trend following strategy
    """
    bull_mu: float = 0.18
    bear_mu: float = -0.77

    bull_sigma: float = 0.184
    bear_sigma: float = 0.184
    constant_sigma: float = 0.184

    bull_lambda: float = 0.36
    bear_lambda: float = 2.53


@dataclass
class BoundaryData(BaseData):
    """
    Buy & Sell Boundaries in trend following strategy
    """

    sell_boundary: float = 0
    buy_boundary: float = 0
