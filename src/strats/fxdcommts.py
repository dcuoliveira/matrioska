import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.portfolio_tools.Diagnostics import Diagnostics
from src.utils.conn_data import load_pickle, save_pickle

fxdcomm_groups = {

    "ALL": ["WDO1", "USDCLP", "USDZAR", "USDAUD", "USDCAD"]

}

sysname = "fxdcommts"

USE_LAST_DATA = False

if __name__ == "__main__":

    if not USE_LAST_DATA:
        input_path = os.path.join(os.getcwd(), "src", "data", "inputs")
        target_dict = load_pickle(os.path.join(input_path, sysname, "{}.pickle".format(sysname)))
        bars_info = target_dict["bars"]
        carry_info = target_dict["carry"]
        signals_info = target_dict["signals"]
        quotes_info = load_pickle(os.path.join(input_path, "quotes.pickle"))

        forecasts_info = {}
        for inst in list(signals_info.keys()):
            tmp_all_signals = signals_info[inst]

            tmp_forecasts = []
            for eq in list(tmp_all_signals.keys()):
                tmp_forecasts.append(pd.DataFrame(np.where(tmp_all_signals[eq][eq] > tmp_all_signals[eq]["upper_band"],
                                                        -1,
                                                        np.where(tmp_all_signals[eq][eq] < tmp_all_signals[eq]["lower_band"],
                                                                1,
                                                                0)), columns=[eq], index=tmp_all_signals[eq][eq].index))
            tmp_forecasts_df = pd.concat(tmp_forecasts, axis=1)

            tmp_forecasts = pd.DataFrame(tmp_forecasts_df.mean(axis=1),
                                        columns=["Close"],
                                        index=tmp_forecasts_df.mean(axis=1).index)
            forecasts_info[inst] = tmp_forecasts

        cerebro = Cerebro(bars=bars_info,
                        forecasts=forecasts_info,
                        carry=carry_info,
                        quotes=quotes_info,
                        groups=fxdcomm_groups)

        portfolio_df = cerebro.run_backtest(instruments=fxdcomm_groups["ALL"],
                                            bar_name="Close",
                                            vol_window=90,
                                            vol_target=0.1,
                                            resample_freq="B",
                                            capital=20000000)

        target_dict["portfolio"] = portfolio_df
        output_path = os.path.join(os.getcwd(), "src", "data", "outputs", sysname, "{}.pickle".format(sysname))
        save_pickle(obj=target_dict, path=output_path)
    else:
        output_path = os.path.join(os.getcwd(), "src", "data", "outputs", sysname, "{}.pickle".format(sysname))
        target_dict = load_pickle(path=output_path)
        portfolio_df = target_dict["portfolio"]

    diagnostics = Diagnostics(portfolio_df=portfolio_df)
    diagnostics.default_metrics(sysname=sysname)
    diagnostics.save_backtests(sysname=sysname)
    diagnostics.save_diagnostics(instruments=fxdcomm_groups["ALL"], sysname=sysname)
