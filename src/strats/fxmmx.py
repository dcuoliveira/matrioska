import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.utils.conn_data import load_pickle, save_pickle

G10 = ["USDEUR", "USDJPY", "USDAUD", "USDNZD", "USDCAD", "USDGBP", "USDCHF", "USDSEK", "USDNOK"]
LATAM = ['WDO1', 'USDCLP', 'USDZAR', 'USDMXN', "USDCOP"]
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

sysname = "fxmmx"

if __name__ == "__main__":

    input_path = os.path.join(os.getcwd(), "src", "data", "inputs")
    target_dict = load_pickle(os.path.join(input_path, "fxmm.pickle"))
    bars_info = target_dict["bars"]
    carry_info = target_dict["carry"]
    signals_info = target_dict["signals"]
    quotes_info = load_pickle(os.path.join(input_path, "quotes.pickle"))

    signals = []
    for inst in fxmm_groups["ALL"]:
        tmp_signals = pd.DataFrame(signals_info[inst].resample("B").last().ffill().mean(axis=1),
                                   columns=["{} signals".format(inst)],
                                   index=signals_info[inst].resample("B").last().ffill().index)
        signals.append(tmp_signals)
    signals_df = pd.concat(signals, axis=1)

    forecastsx_df = signals_df.dropna().copy()

    low_df = forecastsx_df.apply(lambda x: x.nsmallest(3), axis=1).fillna(0)
    low_df[low_df != 0] = -1
    high_df = forecastsx_df.apply(lambda x: x.nlargest(3), axis=1).fillna(0)
    high_df[high_df != 0] = 1

    forecasts_df = high_df.add(low_df, axis=1)
    forecasts_df = forecasts_df[["{} signals".format(inst) for inst in fxmm_groups["ALL"]]]
    forecasts_df.columns = signals_df.columns

    forecasts_info = {}
    for inst in fxmm_groups["ALL"]:
        forecasts_info[inst] = forecasts_df[["{} signals".format(inst)]].rename(columns={"{} signals".format(inst): "{} forecasts".format(inst)})

    cerebro = Cerebro(bars=bars_info,
                      forecasts=forecasts_info,
                      carry=carry_info,
                      quotes=quotes_info,
                      groups=fxmm_groups)

    portfolio_df = cerebro.run_backtest(instruments=fxmm_groups["ALL"],
                                        bar_name="Close",
                                        vol_window=90,
                                        vol_target=0.1,
                                        resample_freq="B",
                                        capital=1000000)

    target_dict["portfolio"] = portfolio_df
    output_path = os.path.join(os.getcwd(), "src", "data", "outputs", "{}.pickle".format(sysname))
    save_pickle(obj=target_dict, path=output_path)