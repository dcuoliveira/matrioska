import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.utils.conn_data import load_pickle, save_pickle

G10 = ["USDEUR", "USDJPY", "USDAUD", "USDNZD", "USDCAD", "USDGBP", "USDCHF", "USDSEK", "USDNOK"]
LATAM = ['USDBRL', 'USDCLP', 'USDZAR', 'USDMXN', "USDCOP"]
C3 = ["USDHUF", "USDPLN", "USDCZK"]
ASIA = ["USDCNH", "USDTWD", "USDINR", "USDKRW"]

fxmm_groups = {

    "ALL": G10 + LATAM + C3 + ASIA,
    "G10": G10,
    "LATAM": LATAM,
    "C3": C3,
    "EM": LATAM + C3,
    "ASIA": ASIA

}

if __name__ == "__main__":

    input_path = os.path.join(os.getcwd(), "src", "data", "inputs", "fxmm.pickle")
    target_dict = load_pickle(input_path)
    bars_info = target_dict["bars"]
    carry_info = target_dict["carry"]
    signals_info = target_dict["signals"]

    bars_info["USDBRL"] = bars_info["WDO1"]
    carry_info["USDBRL"] = carry_info["WDO1"]
    signals_info["USDBRL"] = signals_info["WDO1"]

    forecasts_info = {}
    for inst in list(signals_info.keys()):
        tmp_signals = signals_info[inst].resample("B").last().ffill().mean(axis=1)

        tmp_forecasts = pd.DataFrame(np.where(tmp_signals > 0, tmp_signals, tmp_signals),
                                    columns=["{} forecasts".format(inst)],
                                    index=tmp_signals.index)
        forecasts_info[inst] = tmp_forecasts

    cerebro = Cerebro(bars=bars_info,
                      forecasts=forecasts_info,
                      carry=carry_info,
                      groups=fxmm_groups)

    portfolio_df = cerebro.run_backtest(instruments=fxmm_groups["ALL"],
                                        bar_name="Close",
                                        vol_window=90,
                                        vol_target=0.1,
                                        resample_freq="B",
                                        capital=1000000)

    output_path = os.path.join(os.getcwd(), "src", "data", "outputs", "fxmm.pickle")
    target_dict = save_pickle(object=portfolio_df, path=output_path)