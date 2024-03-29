import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from pandas.tseries.offsets import BDay
import os

from src.utils.conn_data import load_pickle, save_pickle
from settings import INPUT_PATH

class Cerebro(object):
    def __init__(self,
                 strat_metadata: classmethod) -> None:
        self.bars = strat_metadata.bars_info
        self.forecasts = strat_metadata.forecasts_info
        self.carry = strat_metadata.carry_info
        self.quotes = load_pickle(os.path.join(INPUT_PATH, "quotes.pickle"))

        # TODO: generalizar dicionario
        self.groups = {"ALL": strat_metadata.instruments}

        self.metadata = load_pickle(path=os.path.join(INPUT_PATH, "metadata.pickle"))

        if "ALL" in list(self.groups.keys()):
            self.intruments = self.groups["ALL"]
        else:
            self.intruments = self.groups[list(self.groups.keys())[0]]

    def standardize_inputs(self,
                           bar_name: str,
                           vol_window: int,
                           resample_freq: str):

        bars_list = []
        vols_list = []
        rets_list = []
        carrys_list = []
        forecasts_list = []
        quotes_list = []
        for inst in self.intruments:
            tmp_bars = self.bars[inst][[bar_name]].resample(resample_freq).last().ffill()

            tmp_rets = tmp_bars.pct_change()
            tmp_vols = tmp_rets.rolling(window=vol_window).std()

            tmp_forecasts = self.forecasts[inst].resample(resample_freq).last().ffill()
            
            if self.carry is None:
                tmp_carry = tmp_forecasts.copy().rename(columns={"{} forecasts".format(inst): bar_name})
                tmp_carry[bar_name] = 0
            else:
                tmp_carry = self.carry[inst][[bar_name]].resample(resample_freq).last().ffill()

            covert_currency_name = self.metadata.loc[self.metadata["bkt_code"] == inst]["convert_currency"].iloc[0]
            tmp_quotes = self.quotes[covert_currency_name][[bar_name]].resample(resample_freq).last().ffill()

            bars_list.append(tmp_bars.rename(columns={bar_name: "{} close".format(inst)}))
            vols_list.append(tmp_vols.rename(columns={bar_name: "{} daily ret % vol".format(inst)}))
            rets_list.append(tmp_rets.rename(columns={bar_name: "{} ret%".format(inst)}))
            carrys_list.append(tmp_carry.rename(columns={bar_name: "{} carry".format(inst)}))
            forecasts_list.append(tmp_forecasts.rename(columns={bar_name: "{} forecasts".format(inst)}))
            quotes_list.append(tmp_quotes.rename(columns={bar_name: "{} quotes".format(covert_currency_name)}))

        self.bars_df = pd.concat(bars_list, axis=1)
        self.vols_df = pd.concat(vols_list, axis=1)
        self.rets_df = pd.concat(rets_list, axis=1)
        self.carrys_df = pd.concat(carrys_list, axis=1)
        self.forecasts_df = pd.concat(forecasts_list, axis=1)
        self.quotes_df = pd.concat(quotes_list, axis=1)
        self.quotes_df = self.quotes_df.loc[:, ~self.quotes_df.columns.duplicated()].copy()

    def check_available_instruments(self,
                                    instruments: list,
                                    current_dt: pd.DatetimeIndex,
                                    bars_df: pd.DataFrame,
                                    vols_df: pd.DataFrame,
                                    forecasts_df: pd.DataFrame,
                                    carrys_df: pd.DataFrame,
                                    rets_df: pd.DataFrame):
        valid_instruments = []
        for inst in instruments:
            check1 = current_dt in bars_df.loc[:, "{} close".format(inst)].dropna().index
            check2 = current_dt - BDay(1) in bars_df.loc[:, "{} close".format(inst)].dropna().index
            check3 = current_dt in vols_df.loc[:, "{} daily ret % vol".format(inst)].dropna().index
            check4 = current_dt in forecasts_df.loc[:, "{} forecasts".format(inst)].dropna().index
            check5 = current_dt in rets_df.loc[:, "{} ret%".format(inst)].dropna().index
            check6 = current_dt in carrys_df.loc[:, "{} carry".format(inst)].dropna().index

            check7 = forecasts_df.loc[current_dt, "{} forecasts".format(inst)] != 0

            if check1 and check2 and check3 and check4 and check5 and check6 and check7:
                valid_instruments.append(inst)

        return valid_instruments

    def run_backtest(self,
                     instruments: list,
                     bar_name: str,
                     vol_window: int,
                     vol_target: float,
                     capital: float,
                     resample_freq: str,
                     reinvest: bool):

        # standardize dict inputs
        self.standardize_inputs(bar_name=bar_name, vol_window=vol_window, resample_freq=resample_freq)

        # get backtest dates
        backtest_dates = self.forecasts_df.dropna().index
        portfolio_df = pd.DataFrame(index=backtest_dates)
        
        # begin event driven backtest
        first_day = True
        for t in tqdm(backtest_dates, desc="Running Backtest", total=len(backtest_dates)):

            if first_day:
                portfolio_df.loc[t, "capital"] = capital
                portfolio_df.loc[t, "nominal"] = 0
                valid_instruments = instruments.copy()
            else:
                valid_instruments = self.check_available_instruments(instruments,
                                                                     t,
                                                                     self.bars_df,
                                                                     self.vols_df,
                                                                     self.forecasts_df,
                                                                     self.carrys_df,
                                                                     self.rets_df)

            if len(valid_instruments) == 0:
                portfolio_df.loc[t, "capital"] = portfolio_df.loc[t - BDay(1), "capital"] if reinvest else capital

            # compute vol. adjusted positions from forecasts and total nominal exposure
            nominal_total = 0
            for inst in valid_instruments:
                
                if first_day:
                    portfolio_df.loc[t, "{} position units".format(inst)] = 0
                    portfolio_df.loc[t, "{} w".format(inst)] = 0
                    portfolio_df.loc[t, "{} leverage".format(inst)] = 0
                else:
                    # separate all inputs needed for the calculations
                    price_change = self.bars_df.loc[t, "{} close".format(inst)] - self.bars_df.loc[t - BDay(1), "{} close".format(inst)]
                    price = self.bars_df.loc[t, "{} close".format(inst)] 
                    previous_capital = portfolio_df.loc[t - BDay(1), "capital"] if reinvest else capital
                    inst_daily_ret_vol = self.vols_df.loc[t, "{} daily ret % vol".format(inst)]
                    forecast = self.forecasts_df.loc[t, "{} forecasts".format(inst)]

                    # convert price change to local currency when needed
                    currency_to_convert = self.metadata.loc[self.metadata["bkt_code"] == inst]["convert_currency"].iloc[0]
                    convert_factor = self.quotes_df.loc[t, "{} quotes".format(currency_to_convert)]
                    ct_size = float(self.metadata.loc[self.metadata["bkt_code"] == inst]["ct_size"].iloc[0])

                    # compute position vol. target in the local currency and the instrument daily vol. in the local currency as well
                    position_vol_target = (previous_capital / len(valid_instruments)) * vol_target * (1 / np.sqrt(252))
                    inst_daily_price_vol = price * convert_factor * inst_daily_ret_vol * ct_size
                    position_units = forecast * position_vol_target / inst_daily_price_vol 

                    # save position units (e.g. cts, notional in USD etc)
                    portfolio_df.loc[t, "{} position units".format(inst)] = position_units
                    nominal_total += abs(position_units * price * convert_factor * ct_size)

            if not first_day:

                # compute instrument weight exposure (we are going to use them below to compute nominal return)
                for inst in valid_instruments:
                    currency_to_convert = self.metadata.loc[self.metadata["bkt_code"] == inst]["convert_currency"].iloc[0]

                    previous_price = self.bars_df.loc[t - BDay(1), "{} close".format(inst)]
                    previous_convert_factor = self.quotes_df.loc[t - BDay(1), "{} quotes".format(currency_to_convert)]
                    ct_size = float(self.metadata.loc[self.metadata["bkt_code"] == inst]["ct_size"].iloc[0])
                    position_units = portfolio_df.loc[t, "{} position units".format(inst)]

                    nominal_inst = position_units * previous_price * previous_convert_factor * ct_size
                    inst_w = nominal_inst / nominal_total if nominal_total != 0 else 0
                    portfolio_df.loc[t, "{} w".format(inst)] = inst_w

                # compute pnl of the last positions, if any
                pnl = 0
                nominal_ret = 0
                for inst in instruments:
                    previos_positions_units = portfolio_df.loc[t - BDay(1), "{} position units".format(inst)]

                    if (previos_positions_units != 0) and (not pd.isna(previos_positions_units)):
                        price_change = self.bars_df.loc[t, "{} close".format(inst)] - self.bars_df.loc[t - BDay(1), "{} close".format(inst)]

                        # convert price change to local currency when needed
                        currency_to_convert = self.metadata.loc[self.metadata["bkt_code"] == inst]["convert_currency"].iloc[0]
                        convert_factor = self.quotes_df.loc[t, "{} quotes".format(currency_to_convert)]
                        ct_size = float(self.metadata.loc[self.metadata["bkt_code"] == inst]["ct_size"].iloc[0])
                        local_currency_change = price_change * convert_factor * ct_size

                        # compute carry differential and pay/recieve it
                        buy_sell_sign = np.sign(previos_positions_units)
                        carry = self.carrys_df.loc[t, "{} carry".format(inst)]
                        inst_carry_pnl = ((previos_positions_units * (1 + carry * buy_sell_sign)) - previos_positions_units) *  buy_sell_sign

                        # compute pnl
                        inst_pnl = local_currency_change * previos_positions_units + inst_carry_pnl
                        portfolio_df.loc[t, "{} pnl".format(inst)] = inst_pnl

                        pnl += inst_pnl
                        nominal_ret += portfolio_df.loc[t - BDay(1), "{} w".format(inst)] * self.rets_df.loc[t, "{} ret%".format(inst)]

                capital_ret = nominal_ret * portfolio_df.loc[t - BDay(1), "leverage"]
                portfolio_df.loc[t, "capital"] = portfolio_df.loc[t - BDay(1), "capital"] + pnl
                portfolio_df.loc[t, "daily pnl"] = pnl
                portfolio_df.loc[t, "nominal ret"] = nominal_ret
                portfolio_df.loc[t, "capital ret"] = capital_ret 
            
            portfolio_df.loc[t, "nominal"] = nominal_total
            portfolio_df.loc[t, "leverage"] = nominal_total / portfolio_df.loc[t, "capital"]
            first_day = False

        return portfolio_df

    
    def run_group_backtest(self,
                          bar_name: str,
                          vol_window: int,
                          vol_target: float,
                          capital: float):

        portfolios = {}
        for group_name in list(self.groups.keys()):
            portfolio_df = self.run_backtest(instruments=self.groups[group_name],
                                             bar_name=bar_name,
                                             vol_window=vol_window,
                                             vol_target=vol_target,
                                             capital=capital)

            portfolios[group_name] = portfolio_df
        
        return portfolios

