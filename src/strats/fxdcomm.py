import os
import pickle
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro


fxdcomm_groups = {

    "ALL": ["USDBRL", "USDCLP", "USDZAR", "USDAUD", "USDCAD"]

}

if __name__ == "__main__":

    file = open(os.path.join(os.getcwd(), "src", "data", "fxdcomm.pickle"), 'rb')
    target_dict = pickle.load(file)
    bars_info = target_dict["bars"]
    carry_info = target_dict["carry"]
    signals_info = target_dict["signals"]

    bars_info["USDBRL"] = bars_info["WDO1"]
    carry_info["USDBRL"] = carry_info["WDO1"]
    signals_info["USDBRL"] = signals_info["WDO1"]

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
                      signals=signals_info,
                      forecasts=forecasts_info,
                      carry=carry_info,
                      groups=fxdcomm_groups)

    portfolio_df = cerebro.run_backtest(instruments=fxdcomm_groups["ALL"],
                                        bar_name="Close",
                                        vol_window=90,
                                        vol_target=0.1,
                                        resample_freq="B",
                                        capital=1000000)

    fim = 1