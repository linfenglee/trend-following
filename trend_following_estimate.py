from typing import Tuple, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pandas import DataFrame

import plotly
import plotly.graph_objs as go

import tushare as ts
import pandas_datareader.data as web

from trend_following_objects import (
    ANNUAL_DAYS,
    State,
    RegimeInfo,
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

    def set_regime_info(self, regime_info: RegimeInfo) -> None:
        """"""

        self.vt_symbol = regime_info.vt_symbol
        self.bull_p = regime_info.bull_profit
        self.bear_l = regime_info.bear_loss

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
                initial_state = State.Bull
                break
            elif price < bear_price:
                initial_state = State.Bear
                break
        if initial_state == State.Bull:
            bull_date = self.close_array[self.close_array > bull_price].index[0]
            bull_point = self.close_array[self.close_array.index < bull_date].min()
            start_date = self.close_array[self.close_array == bull_point].index[0]
        elif initial_state == State.Bear:
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
            if signal == State.Bull:
                if close_array[idx] / max_s < 1 - self.bear_l:
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
                    signal = State.Bear
                    min_s = close_array[idx]
                else:
                    max_s = max(close_array[idx], max_s)
            elif signal == State.Bear:
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
                    signal = State.Bull
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
            if regime["state"] == State.Bull:
                bull_periods.append(period)
                bull_data.extend(data_batch)
            elif regime["state"] == State.Bear:
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
            if regime["state"] == State.Bull:
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


class EstimationProb(object):
    """"""

    def __init__(
            self, data: DataFrame, price_type: str, para_info: ParameterData
    ):
        """"""

        self.data = data
        self.vt_symbol = para_info.vt_symbol
        self.price_array = self.data[price_type]
        self.dt = 1 / ANNUAL_DAYS

        # Parameter Info
        self.bull_mu = para_info.bull_mu
        self.bear_mu = para_info.bear_mu
        self.bull_sigma = para_info.bull_sigma
        self.bear_sigma = para_info.bear_sigma
        self.const_sigma = para_info.constant_sigma
        self.bull_lambda = para_info.bull_lambda
        self.bear_lambda = para_info.bear_lambda

        self.diff_mu = self.bull_mu - self.bear_mu
        self.sum_lambda = self.bull_lambda + self.bear_lambda

    def __str__(self):
        """"""
        return f"Estimation for Current {self.vt_symbol} Bull Probability"

    def set_para_info(self, para_info: ParameterData):
        """"""

        # Parameter Info
        self.bull_mu = para_info.bull_mu
        self.bear_mu = para_info.bear_mu
        self.bull_sigma = para_info.bull_sigma
        self.bear_sigma = para_info.bear_sigma
        self.const_sigma = para_info.constant_sigma
        self.bull_lambda = para_info.bull_lambda
        self.bear_lambda = para_info.bear_lambda

        self.diff_mu = self.bull_mu - self.bear_mu
        self.sum_lambda = self.bull_lambda + self.bear_lambda

    def calc_gt(self, pt: float) -> float:
        """"""

        gt_1 = -self.sum_lambda * pt + self.bear_lambda
        coef_dlns = self.diff_mu * pt * (1 - pt) / np.power(self.const_sigma, 2)
        adjust_term = (self.diff_mu * pt + self.bear_mu - np.power(self.const_sigma, 2) / 2)
        gt_2 = -coef_dlns * adjust_term
        gt = gt_1 + gt_2
        return gt

    def update_pt(self, pt: float, dlns: float) -> float:
        """"""

        coef_dt = self.calc_gt(pt)
        coef_dlns = self.diff_mu * pt * (1 - pt) / np.power(self.const_sigma, 2)
        new_pt = pt + coef_dt * self.dt + coef_dlns * dlns
        update_pt = min(max(new_pt, 0), 1)

        return update_pt

    def estimate_pts(self, start_date: datetime, start_p: float) -> DataFrame:
        """"""

        lns = np.log(self.price_array[self.price_array.index >= start_date])
        dts = lns.index.to_numpy()
        pts = pd.DataFrame(columns=["pt", "dlns"])
        pt = start_p
        for i in range(1, len(dts), 1):
            pre_dt, dt = dts[i-1], dts[i]
            dlns = lns[dt] - lns[pre_dt]
            new_pt = self.update_pt(pt, dlns)
            pts.loc[dt] = [new_pt, dlns]
            pt = new_pt

        return pts

    @staticmethod
    def pts_plot(pt_data: DataFrame) -> None:
        """"""

        trace1 = go.Scatter(
            x=pt_data.index,
            y=pt_data["pt"],
            mode="lines",
            name="pt"
        )

        data = [trace1]
        layout = go.Layout(
            legend={"x": 1, "y": 1},
            title="Buy Region & Sell Region"
        )
        fig = go.Figure(data=data, layout=layout)

        plotly.offline.plot(fig, filename='pt_process.html')


class EstimationEngine(object):
    """"""

    def __init__(self, source: str, vt_symbol: str, price_type: str):
        """"""

        if source == "tushare":
            data = self.get_data_tushare(vt_symbol)
        else:
            data = self.get_data_yahoo(vt_symbol)

        regime_info = RegimeInfo(vt_symbol)
        self.est_para = EstimationParameter(data, price_type, regime_info)

        para_info = ParameterData(vt_symbol)
        self.est_prob = EstimationProb(data, price_type, para_info)

    @staticmethod
    def get_data_tushare(vt_symbol: str) -> DataFrame:
        """"""
        dt = datetime.now()
        start_year = dt.year - 10
        end_dt = dt.strftime("%Y%m%d")
        start_dt = dt.replace(year=start_year).strftime("%Y%m%d")

        pro = ts.pro_api('c3450fdabd780ea8ef13ebdd0230c2e52a5737b05cfb0377eecc2f84')
        df = pro.daily(ts_code=vt_symbol, start_date=start_dt, end_date=end_dt)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df.sort_values(by='trade_date', inplace=True)
        df.set_index("trade_date", inplace=True)
        df['log_rtn'] = np.log(df['close']).diff()

        return df

    @staticmethod
    def get_data_yahoo(vt_symbol: str) -> DataFrame:
        """"""

        end_dt = datetime.now()
        start_year = end_dt.year - 10
        start_dt = end_dt.replace(year=start_year)

        df = web.get_data_yahoo(vt_symbol, start_dt, end_dt)
        df.index = pd.to_datetime(df.index)
        df.drop(columns=["Volume"], inplace=True)
        df.columns = ['open', 'high', 'low', 'close', 'adjust_close']
        df['log_rtn'] = np.log(df['adjust_close']).diff()

        return df

    def set_regime_info(self, regime_info: RegimeInfo) -> None:
        """"""

        self.est_para.set_regime_info(regime_info)

    def set_para_info(self, para_info: ParameterData) -> None:
        """"""

        self.est_prob.set_para_info(para_info)

    def estimate_para(self):
        """"""

        est_para_info = self.est_para.get_para_info()
        return est_para_info

    def estimate_prob(self, start_dt: datetime, start_pt: float) -> DataFrame:
        """"""

        est_pts = self.est_prob.estimate_pts(start_dt, start_pt)
        return est_pts

    def main_est(
            self,
            regime_info: RegimeInfo,
            start_dt: datetime,
            start_pt: float
    ) -> Tuple:
        """"""

        self.set_regime_info(regime_info)

        est_para_info = self.estimate_para()

        self.set_para_info(est_para_info)

        est_pts = self.estimate_prob(start_dt, start_pt)

        return est_para_info, est_pts


if __name__ == '__main__':
    """"""

    data_source = "tushare"
    ts_code = "000001.SZ"
    price_type = "close"
    start_date = datetime(2011, 10, 1)
    start_prob = 0

    regime_data = RegimeInfo(ts_code, 0.3, 0.3)

    est_engine = EstimationEngine(data_source, ts_code, price_type)
    est_engine.set_regime_info(regime_data)
    para_data = est_engine.estimate_para()
    est_engine.set_para_info(para_data)
    pts = est_engine.estimate_prob(start_date, start_prob)




