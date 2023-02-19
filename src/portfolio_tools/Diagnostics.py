import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from settings import OUTPUT_PATH
from src.utils.conn_data import save_pickle
from src.viz.data_viz import usdollars_format

plt.style.use("bmh")

class Diagnostics(object):
    def __init__(self,
                 portfolio_df: pd.DataFrame) -> None:
        self.portfolio_df = portfolio_df

    def default_metrics(self,
                        sysname):
        
        if "capital ret" in self.portfolio_df.columns:
            self.portfolio_df["cum ret"] = (1 + self.portfolio_df["capital ret"]).cumprod()
            self.portfolio_df["drawdown"] = self.portfolio_df["cum ret"] / self.portfolio_df["cum ret"].cummax() - 1
            self.sharpe = self.portfolio_df["capital ret"].mean() / self.portfolio_df["capital ret"].std() * np.sqrt(253)
            self.drawdown_max = self.portfolio_df["drawdown"].min() * 100
            self.vol_ann = self.portfolio_df["capital ret"].std() * np.sqrt(252) * 100

            self.summary = "{}: \nSharpe: {} \nDrawdown: {}\nVolatility: {}\n".format(
                sysname.upper(), round(self.sharpe, 2), round(self.drawdown_max, 1), round(self.vol_ann, 1)
            )    
        else:
            print('There is no columns named "capital ret" in the "portfolio_df" object')

        if "daily pnl" in self.portfolio_df.columns:
            self.portfolio_df["cum pnl"] = self.portfolio_df["daily pnl"].cumsum()
            self.portfolio_df["drawdown"] = self.portfolio_df["cum pnl"] / self.portfolio_df["cum pnl"].cummax()
            self.money_sharpe = self.portfolio_df["daily pnl"].mean() / self.portfolio_df["daily pnl"].std() * np.sqrt(252)
            self.money_drawdown_max = self.portfolio_df["drawdown"].replace(np.inf, np.nan).replace(-np.inf, np.nan).min()
            self.money_vol_daily = self.portfolio_df["daily pnl"].std()

            self.money_summary = "{}: \nSharpe: {} \nDrawdown: {}\nDaily Volatility: {}\n".format(
                sysname.upper(), round(self.money_sharpe, 2), usdollars_format(self.money_drawdown_max), usdollars_format(self.money_vol_daily)
            )    
        else:
            print('There is no columns named "daily pnl" in the "portfolio_df" object')

    def save_backtests(self,
                       sysname):

        ax = sns.lineplot(data=self.portfolio_df["cum ret"], linewidth=1.5, palette="deep")
        ax.annotate(
            self.summary,
            xy=(0.2, 0.8),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.3),
            ha="center",
            va="center",
            family="serif",
            size="8"
        )
        plt.title("Cumulative Returns")
        plt.savefig("{path}/{sysname}/{sysname}_cumret.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

        self.portfolio_df.to_excel("{path}/{sysname}/{sysname}_portfolio_info.xlsx".format(path=OUTPUT_PATH, sysname=sysname)) 
        save_pickle(path="{path}/{sysname}/{sysname}_portfolio_info.obj".format(path=OUTPUT_PATH, sysname=sysname), obj=self.portfolio_df)  

        ax = sns.lineplot(data=self.portfolio_df["cum pnl"], linewidth=1.5, palette="deep")
        ax.annotate(
            self.money_summary,
            xy=(0.2, 0.8),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.3),
            ha="center",
            va="center",
            family="serif",
            size="8"
        )
        plt.title("Cumulative pnl")
        ylabels = [usdollars_format(x) for x in ax.get_yticks()]
        ax.set_yticklabels(ylabels)
        plt.savefig("{path}/{sysname}/{sysname}_cumpnl.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

    def save_diagnostics(self,
                         instruments,
                         sysname):
        for inst in instruments:
            self.portfolio_df["{} w".format(inst)].fillna(0).plot()
        plt.title("Instrument Weights")
        plt.savefig("{path}/{sysname}/{sysname}_weights.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

        self.portfolio_df["leverage"].plot()
        plt.title("Portfolio Leverage")
        plt.savefig("{path}/{sysname}/{sysname}_leverage.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

        plt.scatter(self.portfolio_df.index, self.portfolio_df["capital ret"] * 100)
        plt.title("Daily Return Scatter Plot")
        plt.savefig("{path}/{sysname}/{sysname}_scatter.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

        ax = sns.histplot(data=self.portfolio_df["daily pnl"], palette="deep")
        ax.annotate(
            "{}: \nAvg. PnL: {} \n1 Std. PnL: {}\n".format(
                sysname.upper(), usdollars_format(self.portfolio_df["daily pnl"].mean()), usdollars_format(self.portfolio_df["daily pnl"].std())
            ),
            xy=(0.2, 0.8),
            xycoords="axes fraction",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.3),
            ha="center",
            va="center",
            family="serif",
            size="8"
        )
        xlabels = [usdollars_format(x) for x in ax.get_xticks()]
        ax.set_xticklabels(xlabels, rotation=40)
        plt.title("Daily PnL")
        plt.savefig("{path}/{sysname}/{sysname}_hist_daily_pnl.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()

        ax = sns.lineplot(data=self.portfolio_df["daily pnl"].rolling(window=30).std(), linewidth=1.5, palette="deep")
        ylabels = [usdollars_format(x) for x in ax.get_yticks()]
        ax.set_yticklabels(ylabels)
        plt.title("Daily PnL Vol.")
        plt.savefig("{path}/{sysname}/{sysname}_daily_pnl_vol.png".format(path=OUTPUT_PATH, sysname=sysname), bbox_inches="tight")
        plt.close()