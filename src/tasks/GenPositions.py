def run_simulation(instruments,
                   historical_data,
                   portfolio_vol,
                   subsystems_dict,
                   debug=True):
    
    test_ranges = []
    for subsystem in subsystems_dict.keys():
        test_ranges.append(subsystems_dict[subsystem]["strat_df"].index)
    start = max(test_ranges, key=lambda x:[0])[0]
    print(start) #start running the combined strategy from 2012-02-09 onwards, since that is when all the 3 strategy data is available

    portfolio_df = pd.DataFrame(index=historical_data[start:].index).reset_index()
    portfolio_df.loc[0, "capital"] = 10000


    """
    Run Simulation
    """
    for i in portfolio_df.index:
        date = portfolio_df.loc[i, "date"]
        strat_scalar = 2 #strategy scalar (refer to post)
        """
        Get PnL, Scalars
        """
        if i != 0:
            date_prev = portfolio_df.loc[i - 1 ,"date"]
            pnl, nominal_ret = backtest_utils.get_backtest_day_stats(portfolio_df, instruments, date, date_prev, i, historical_data)
            #Obtain strategy scalar
            strat_scalar = backtest_utils.get_strat_scaler(portfolio_df, lookback=100, vol_target=portfolio_vol, idx=i, default=strat_scalar)
            #now, our strategy leverage / scalar should dynamically equilibriate to achieve target exposure, we see that in fact this is the case!

        portfolio_df.loc[i, "strat scalar"] = strat_scalar

        """
        Get Positions
        """
        inst_units = {}
        for inst in instruments:
            inst_dict = {}
            for subsystem in subsystems_dict.keys():
                subdf = subsystems_dict[subsystem]["strat_df"]
                subunits = subdf.loc[date, "{} units".format(inst)] if "{} units".format(inst) in subdf.columns and date in subdf.index  else 0
                subscalar = portfolio_df.loc[i, "capital"] / subdf.loc[date, "capital"] if date in subdf.index else 0
                inst_dict[subsystem] = subunits * subscalar
            inst_units[inst] = inst_dict

        nominal_total = 0            
        for inst in instruments:
            combined_sizing = 0
            for subsystem in subsystems_dict.keys():
                combined_sizing += inst_units[inst][subsystem] * subsystems_config[subsystem]
            position = combined_sizing * strat_scalar
            portfolio_df.loc[i, "{} units".format(inst)] = position
            if position != 0:
                nominal_total += abs(position * backtest_utils.unit_dollar_value(inst, historical_data, date))
        
        for inst in instruments:
            units = portfolio_df.loc[i, "{} units".format(inst)]
            if units != 0:
                nominal_inst = units * backtest_utils.unit_dollar_value(inst, historical_data, date)
                inst_w = nominal_inst / nominal_total
                portfolio_df.loc[i, "{} w".format(inst)] = inst_w
            else:
                portfolio_df.loc[i, "{} w".format(inst)] = 0

        nominal_total = backtest_utils.set_leverage_cap(portfolio_df, instruments, date, i, nominal_total, 10, historical_data)

        """
        Perform Calculations for Date
        """
        portfolio_df.loc[i, "nominal"] = nominal_total
        portfolio_df.loc[i, "leverage"] = nominal_total / portfolio_df.loc[i, "capital"]
        if True: print(portfolio_df.loc[i])    
    
    portfolio_df.set_index("date", inplace=True)

    diagnostic_utils.save_backtests(
        portfolio_df=portfolio_df, instruments=instruments, brokerage_used=brokerage_used, sysname="HANGUKQUANT"
    )
    diagnostic_utils.save_diagnostics(
        portfolio_df=portfolio_df, instruments=instruments, brokerage_used=brokerage_used, sysname="HANGUKQUANT"
    )

    return portfolio_df

def get_subsys_pos(sysname):
    pass

if __name__ == "__main__":

    from src.strats.fxdcommts import fxdcomm_groups
    from src.strats.fxmmts import fxmm_groups
    from src.strats.ratesmmts import ratesmm_groups

    VOL_TARGET = 0.1
    strats = {
        
        "fxdcommts": 0.25, 
        "fxmmts": 0.25,
        "fxmmx": 0.25,
        "ratesmmts": 0.25

                }
    
    subsystems_config = portfolio_config["subsystems"][brokerage_used]
    strats = {}
    for subsystem in subsystems_config.keys():
        if subsystem == "lbmom":
            strat = Lbmom(
                instruments_config=portfolio_config["instruments_config"][subsystem][brokerage_used], 
                historical_df=historical_data, 
                simulation_start=sim_start, 
                vol_target=VOL_TARGET, 
                brokerage_used=brokerage_used
            )
        elif subsystem == "lsmom":
            strat = Lsmom(
                instruments_config=portfolio_config["instruments_config"][subsystem][brokerage_used], 
                historical_df=historical_data, 
                simulation_start=sim_start, 
                vol_target=VOL_TARGET, 
                brokerage_used=brokerage_used
            )
        elif subsystem == "skprm":
            strat = Skprm(
                instruments_config=portfolio_config["instruments_config"][subsystem][brokerage_used], 
                historical_df=historical_data, 
                simulation_start=sim_start, 
                vol_target=VOL_TARGET, 
                brokerage_used=brokerage_used
            )
        else:
            pass#...
        strats[subsystem] = strat
    
    subsystems_dict = {}
    instruments = []
    for k, v in strats.items():
        strat_df, strat_inst = get_subsys_pos(sysname=k) #see if you want to print the items
        subsystems_dict[k] = {
            "strat_df": strat_df,
            "strat_inst": strat_inst
        }
        instruments += strat_inst
    instruments = list(set(instruments))

    portfolio_df = run_simulation(intruments=instruments,
                                  historical_data=historical_data,
                                  portfolio_vol=VOL_TARGET,
                                  subsystems_dict=subsystems_dict,
                                  debug=True)