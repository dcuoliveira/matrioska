import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.utils.conn_data import load_pickle, save_pickle

ALL = ['FV1', 'TU1', 'IK1', 'OAT1', 'DU1', 'G 1', 'JB1', 'XM1', 'YM1', 'CN1', 'RX1', 'OE1', 'TY1']

ratesmm_groups = {

    "ALL": ALL,

}

if __name__ == "__main__":

    input_path = os.path.join(os.getcwd(), "src", "data", "inputs")
    target_dict = load_pickle(os.path.join(input_path, "ratesmm.pickle"))
    bars_info = target_dict["bars"]
    signals_info = target_dict["signals"]["signals"]
    quotes_info = load_pickle(os.path.join(input_path, "quotes.pickle"))

    forecasts_info = {}
    for inst in ratesmm_groups["ALL"]:
        tmp_signals = signals_info[inst].resample("B").last().ffill().mean(axis=1)

        # zscore
        tmp_signals = (tmp_signals - tmp_signals.rolling(window=252).mean()) / tmp_signals.rolling(window=252).std()

        tmp_forecasts = pd.DataFrame(np.where(tmp_signals > 0.5, tmp_signals, np.where(tmp_signals < -0.5, tmp_signals, 0)),
                                    columns=["{} forecasts".format(inst)],
                                    index=tmp_signals.index)
        forecasts_info[inst] = tmp_forecasts

    cerebro = Cerebro(bars=bars_info,
                      forecasts=forecasts_info,
                      carry=None,
                      quotes=quotes_info,
                      groups=ratesmm_groups)

    portfolio_df = cerebro.run_backtest(instruments=ratesmm_groups["ALL"],
                                        bar_name="Close",
                                        vol_window=90,
                                        vol_target=0.1,
                                        resample_freq="B",
                                        capital=1000000)

    output_path = os.path.join(os.getcwd(), "src", "data", "outputs", "ratesmmts.pickle")
    target_dict = save_pickle(object=portfolio_df, path=output_path)