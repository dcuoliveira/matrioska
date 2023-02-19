import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from settings import OUTPUT_PATH
from src.utils.conn_data import save_pickle

plt.style.use("bmh")

class Diagnostics(object):
    def __init__(self,
                 portfolio_df: pd.DataFrame) -> None:
        self.portfolio_df = portfolio_df

    def default_metrics(self,
                        sysname):
        
        if "capital ret" in self.portfolio_df.columns:
            tmp_portfolio_df = self.portfolio_df[["capital ret"]]
            tmp_portfolio_df["cum ret"] = (1 + tmp_portfolio_df["capital ret"]).cumprod()
            tmp_portfolio_df["drawdown"] = tmp_portfolio_df["cum ret"] / tmp_portfolio_df["cum ret"].cummax() - 1
            self.sharpe = tmp_portfolio_df["capital ret"].mean() / tmp_portfolio_df["capital ret"].std() * np.sqrt(253)
            self.drawdown_max = tmp_portfolio_df["drawdown"].min() * 100
            self.volatility = tmp_portfolio_df["capital ret"].std() * np.sqrt(253) * 100 #annualised percent vol

            self.summary = "{}: \nSharpe: {} \nDrawdown: {}\nVolatility: {}\n".format(
                sysname, round(self.sharpe, 2), round(self.drawdown_max, 2), round(self.vol_ann, 2)
            )    
        else:
            print('There is no columns named "capital ret" in the "portfolio_df" object')

    def save_backtests(self,
                       sysname):

        ax = sns.lineplot(data=self.portfolio_df["cum ret"], linewidth=2.5, palette="deep")
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
        plt.savefig("{}/{}_cumret.png".format(OUTPUT_PATH, sysname), bbox_inches="tight")
        plt.close()

        self.portfolio_df.to_excel("{}/{}_portfolio_info.xlsx".format(OUTPUT_PATH, sysname)) 
        save_pickle(path="{}/{}_portfolio_info.obj".format(OUTPUT_PATH, sysname), obj=self.portfolio_df)    

    def save_diagnostics(self,
                         instruments,
                         sysname):
        for inst in instruments:
            self.portfolio_df["{} w".format(inst)].plot()
        plt.title("Instrument Weights")
        plt.savefig("{}/{}_weights.png".format(OUTPUT_PATH, sysname), bbox_inches="tight")
        plt.close()

        self.portfolio_df["leverage"].plot()
        plt.title("Portfolio Leverage")
        plt.savefig("{}/{}_leverage.png".format(OUTPUT_PATH, sysname), bbox_inches="tight")
        plt.close()

        plt.scatter(self.portfolio_df.index, self.portfolio_df["capital ret"] * 100)
        plt.title("Daily Return Scatter Plot")
        plt.savefig("{}/{}_scatter.png".format(OUTPUT_PATH, sysname), bbox_inches="tight")
        plt.close()

        #histogram plot, etc etc