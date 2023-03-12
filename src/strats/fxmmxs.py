import os
import pandas as pd
import numpy as np

from src.portfolio_tools.Cerebro import Cerebro
from src.portfolio_tools.Diagnostics import Diagnostics
from src.utils.conn_data import load_pickle, save_pickle
from settings import INPUT_PATH, OUTPUT_PATH

class fxmmxs:
    def __init__(self, simulation_start, vol_target) -> None:
        self.sysname = "fxmmxs"
        G10 = ["USDEUR", "USDJPY", "USDAUD", "USDNZD", "USDCAD", "USDGBP", "USDCHF", "USDSEK", "USDNOK"]
        LATAM = ['WDO1', 'USDCLP', 'USDZAR', 'USDMXN', "USDCOP"]
        C3 = ["USDHUF", "USDPLN", "USDCZK"]
        ASIA = ["USDCNH", "USDTWD", "USDINR", "USDKRW"]
        self.instruments = G10 + LATAM + C3 + ASIA
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
        signals = []
        for inst in self.instruments:
            tmp_signals = pd.DataFrame(self.signals_info[inst].resample("B").last().ffill().mean(axis=1),
                                    columns=["{} signals".format(inst)],
                                    index=self.signals_info[inst].resample("B").last().ffill().index)
            signals.append(tmp_signals)
        signals_df = pd.concat(signals, axis=1)

        forecastsx_df = signals_df.dropna().copy()

        low_df = forecastsx_df.apply(lambda x: x.nsmallest(3), axis=1).fillna(0)
        low_df[low_df != 0] = -1
        high_df = forecastsx_df.apply(lambda x: x.nlargest(3), axis=1).fillna(0)
        high_df[high_df != 0] = 1

        forecasts_df = high_df.add(low_df, axis=1)
        forecasts_df = forecasts_df[["{} signals".format(inst) for inst in self.instruments]]
        forecasts_df.columns = signals_df.columns

        forecasts_info = {}
        for inst in self.instruments:
            forecasts_info[inst] = forecasts_df[["{} signals".format(inst)]].rename(columns={"{} signals".format(inst): "{} forecasts".format(inst)})
        
        return forecasts_info

USE_LAST_DATA = False

if __name__ == "__main__":
    strat_metadata = fxmmxs(simulation_start=None, vol_target=0.2)

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