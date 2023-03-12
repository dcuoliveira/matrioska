import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.portfolio_tools.Diagnostics import Diagnostics
from src.utils.conn_data import load_pickle, save_pickle
from settings import INPUT_PATH, OUTPUT_PATH

class fxdcommts:
    def __init__(self, simulation_start, vol_target) -> None:
        self.sysname = "fxdcommts"
        self.instruments = ["WDO1", "USDCLP", "USDCOP", "USDZAR", "USDAUD", "USDCAD"] # "EURNOK"
        self.simulation_start = simulation_start
        self.vol_target = vol_target

        # inputs
        self.strat_inputs = load_pickle(os.path.join(INPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        self.bars_info = self.strat_inputs["bars"]
        self.carry_info = self.strat_inputs["carry"]
        self.signals_info = self.strat_inputs["signals"]
        self.forecasts_info = self.build_forecasts()

        # outputs
        if os.path.exists(os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname))):
            self.strat_outputs = load_pickle(path=os.path.join(OUTPUT_PATH, self.sysname, "{}.pickle".format(self.sysname)))
        else:
            self.strat_outputs = None

    def build_forecasts(self):
        forecasts_info = {}
        for inst in list(self.signals_info.keys()):
            tmp_all_signals = self.signals_info[inst]

            tmp_forecasts = []
            for eq in list(tmp_all_signals.keys()):
                tmp_forecasts.append(pd.DataFrame(np.where(tmp_all_signals[eq][eq] > tmp_all_signals[eq]["upper_band"],
                                                        -1,
                                                        np.where(tmp_all_signals[eq][eq] < tmp_all_signals[eq]["lower_band"],
                                                                1,
                                                                0)), columns=[eq], index=tmp_all_signals[eq][eq].index))
            tmp_forecasts_df = pd.concat(tmp_forecasts, axis=1)

            tmp_forecasts = pd.DataFrame(np.where(tmp_forecasts_df.sum(axis=1) > 0, 1, np.where(tmp_forecasts_df.sum(axis=1) < 0, -1, 0)),
                                         columns=["Close"],
                                         index=tmp_forecasts_df.sum(axis=1).index)
            forecasts_info[inst] = tmp_forecasts
        
        return forecasts_info


USE_LAST_DATA = True

if __name__ == "__main__":
    strat_metadata = fxdcommts(simulation_start=None, vol_target=0.2)

    if not USE_LAST_DATA:
        cerebro = Cerebro(strat_metadata=strat_metadata)

        portfolio_df = cerebro.run_backtest(instruments=strat_metadata.instruments,
                                            bar_name="Close",
                                            vol_window=90,
                                            vol_target=strat_metadata.vol_target,
                                            resample_freq="B",
                                            capital=10000,
                                            reinvest=False)

        strat_metadata.strat_inputs["portfolio"] = portfolio_df
        output_path = os.path.join(OUTPUT_PATH, strat_metadata.sysname, "{}.pickle".format(strat_metadata.sysname))
        save_pickle(obj=strat_metadata.strat_inputs, path=output_path)
    else:
        output_path = os.path.join(OUTPUT_PATH, strat_metadata.sysname, "{}.pickle".format(strat_metadata.sysname))
        target_dict = load_pickle(path=output_path)
        portfolio_df = target_dict["portfolio"]

    diagnostics = Diagnostics(portfolio_df=portfolio_df)
    diagnostics.default_metrics(sysname=strat_metadata.sysname)
    diagnostics.save_backtests(sysname=strat_metadata.sysname)
    diagnostics.save_diagnostics(instruments=strat_metadata.instruments, sysname=strat_metadata.sysname)
