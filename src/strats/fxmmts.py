import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.portfolio_tools.Diagnostics import Diagnostics
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

sysname = "fxmmts"

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
            tmp_signals = signals_info[inst].resample("B").last().ffill().mean(axis=1)

            tmp_forecasts = pd.DataFrame(np.where(tmp_signals > 0, 1, -1),
                                        columns=["{} forecasts".format(inst)],
                                        index=tmp_signals.index)
            forecasts_info[inst] = tmp_forecasts

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
                                            capital=35000000,
                                            reinvest=False)

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
    diagnostics.save_diagnostics(instruments=fxmm_groups["ALL"], sysname=sysname)