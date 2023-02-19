import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import dataframe_image as dfi
 
from statsmodels.tsa.stattools import ccf
import statsmodels.api as sm
 
from src.settings import OUTPUTS_PATH
 
plt.style.use('bmh')
 
############################################################################## PLOTS

def usdollars_format(value):
    num = float('{:.3g}'.format(value))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return 'US$ {}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def reais_format(value):
    num = float('{:.3g}'.format(value))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return 'R$ {}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
 
 
def plot_buysell_markers(bars,
                         signal,
                         positions,
                         fig_title,
                         start_date,
                         end_date):
    positions_sign = pd.DataFrame(np.where(positions > 0,
                                     1,
                                     np.where(positions < 0,
                                              -1,
                                              0)),
                                  columns=["signal"],
                                  index=positions.index)
 
    out = pd.concat([bars, signal, positions_sign, positions_sign], axis=1).loc[start_date:end_date]
    out.columns = ["price", "signal", "position sign", "position sign diff"]
   
    arrows = []
    comecar = 0
    for index, row in out.iterrows():
        current = out.loc[index]["position sign diff"]
        if comecar == 0:
            last = current
            comecar = 1
            continue
 
        if current == last:
            arrow_side = 0
        elif (current == 1) and (last == -1):
            arrow_side = 2
        elif (current == -1) and (last == 1):
            arrow_side = -2
        elif (current == 0) and (last == 1):
            arrow_side = 3
        elif (current == 0) and (last == -1):
            arrow_side = -3
        elif (current == 1) and (last == 0):
            arrow_side = 4
        elif (current == -1) and (last == 0):
            arrow_side = -4
        else:
            raise Exception("Erro")
 
        last = current
 
        out.loc[index, "arrow"] = arrow_side
 
    # Plot two charts to assess trades and equity curve
    fig, (ax1, ax2) = plt.subplots(2,
                                   sharex=False,
                                   figsize=(18, 13),
                                   gridspec_kw={'height_ratios': [3, 1]})
 
    ax1.plot(out.index,
             out["price"],
             color="blue")
   
    # virada de mao pra comprar
    ax1.plot(out.loc[out.arrow == 2.0].index,
             out.price[out.arrow == 2.0],
             '^',
             markersize=10,
             color='green')
   
    # virada de mao pra venda
    ax1.plot(out.loc[out.arrow == -2.0].index,
             out.price[out.arrow == -2.0],
             'v',
             markersize=10,
             color='red')
   
    # zerada de compra
    ax1.plot(out.loc[out.arrow == 3.0].index,
             out.price[out.arrow == 3.0],
             'o',
             markersize=5,
             color='green')
   
    # nova venda
    ax1.plot(out.loc[out.arrow == -3.0].index,
             out.price[out.arrow == -3.0],
             'o',
             markersize=5,
            color='red')
   
    # nova compra
    ax1.plot(out.loc[out.arrow == 4.0].index,
             out.price[out.arrow == 4.0],
             '^',
             markersize=10,
             color='green')
   
    # zerada de venda
    ax1.plot(out.loc[out.arrow == -4.0].index,
             out.price[out.arrow == -4.0],
             'v',
             markersize=10,
             color='red')
   
    # add signal
    ax2.plot(out.index,
             out["signal"],
             color="red")
   
    ax1.set_title(fig_title)
   
    
    return out
 
 
def plot_multiaxis(data,
                   name_to_hightlight="nan",
                   start_date="2017-01-01",
                   end_date="2020-01-01",
                   force_fix_second_axis=False,
                   colors=['black', 'red', 'royalblue', 'limegreen', 'mediumorchid', 'orange'],
                   labels_col = "Use column name",
                   annotate=False,
                   save_fig = False,
                   path_savefig = OUTPUTS_PATH,
                   filename = "fig1"):
 
    fig, ax_orig = plt.subplots(figsize=(15, 10))
    time_series = data.loc[start_date:end_date].ffill()
    standardized_data = pd.DataFrame()
   
    if labels_col == "Use column name":
        labels = list(data.columns)
    else:
        labels = labels_col
 
    if name_to_hightlight == "nan":
        alpha = 1
    else:
        alpha = 0.1
   
    y_pos = []
    for i, (color, label) in enumerate(zip(colors, labels)):
 
        standardized_data[time_series.columns[i]] = (time_series[time_series.columns[i]] - time_series[time_series.columns[i]].min()) / \
            (time_series[time_series.columns[i]].max() - time_series[time_series.columns[i]].min())
 
        if i == 0:
            ax = ax_orig
        else:
            ax = ax_orig.twinx()
            ax.spines['right'].set_position(('outward', 50 * (i - 1)))
       
        if label == name_to_hightlight:
            ax.plot(time_series.index,time_series[time_series.columns[i]], color=color, alpha=1)
        else:
            ax.plot(time_series.index,time_series[time_series.columns[i]], color=color, alpha=alpha)
 
       
        y_pos.append(standardized_data[time_series.columns[i]][-1])
        if annotate:
 
            if (len(y_pos) > 1) and (y_pos[-2] - 0.04 < y_pos[-1] < y_pos[-2] + 0.04):
 
                tilt = 0.04 * ((time_series[time_series.columns[i]].max() - time_series[time_series.columns[i]].min()) + time_series[time_series.columns[i]].min())
                if y_pos[-1] + 0.04 > 1:
                    ax.text(time_series.index[-1],
                            time_series[time_series.columns[i]][-1] - tilt,
                            "{:10.2f}".format(time_series[time_series.columns[i]][-1]),
                            ha="center", va="center", size=10,
                            bbox=dict(boxstyle="round, pad=0.3", fc="white", ec=color, lw=2),
                            visible=True)
                else:
                    ax.text(time_series.index[-1],
                            time_series[time_series.columns[i]][-1] + tilt,
                            "{:10.2f}".format(time_series[time_series.columns[i]][-1]),
                            ha="center", va="center", size=10,
                            bbox=dict(boxstyle="round, pad=0.3", fc="white", ec=color, lw=2),
                            visible=True)
 
            else:
                ax.text(time_series.index[-1],
                        time_series[time_series.columns[i]][-1],
                        "{:10.2f}".format(time_series[time_series.columns[i]][-1]),
                        ha="center", va="center", size=10,
                        bbox=dict(boxstyle="round, pad=0.3", fc="white", ec=color, lw=2),
                        visible=True)
 
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis='y', colors=color)
        ax.set_title('')
 
    if save_fig:
        fig_path = path_savefig + '\\' + filename + '.jpg'
        plt.savefig(fig_path, dpi = 200, bbox_inches='tight')
   
    return fig
 
 
def plot_ts_heatmap(df,
                    color_map="coolwarm",
                    title='Ranking dos sinais agregados para cada moeda \n\n\n',
                    annot=False, folder_name="FXMM", size = 200):
 
    # heatmap
    fig, ax = plt.subplots(figsize=(20, 10))# plot heatmap
    sns.heatmap(df,
               cmap=color_map,
               vmin=df.min().min(),
               vmax=df.max().max(),
               square=True,
               linewidth=0.3,
               cbar_kws={"shrink": .8},
               annot=annot)
    ax.xaxis.tick_top()
    xticks_labels = df.columns
    plt.xticks(np.arange(len(df.columns)) + .6, labels=xticks_labels)# axis labels
    plt.xlabel('')
    plt.ylabel('')
    title = title.upper()
    plt.title(title, loc='left')
    plt.savefig(OUTPUTS_PATH + "\\"+ folder_name + "\\"+ title + '.jpg', dpi = size)
 
 
def plot_hist_compare_two_money_axis(df):
    bins = np.linspace(df.min().min(), df.max().max(), 30)
    x = df[df.columns[0]]
    y = df[df.columns[1]]
    fig, ax = plt.subplots()
    ax.hist([x, y],
            bins,
            label=[df.columns[0], df.columns[1]])
    ax.legend(loc='upper right')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(reais_format))
 
 
def plot_mts_money_axis(df):
    dates = df.index
    fig, ax = plt.subplots()
 
    for colname in list(df.columns):
        x = df[colname]
        ax.plot(dates,
                x,
                label=colname)
 
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(reais_format))
 
 
def plot_ccf(x,
             y,
             adjusted,
             confidence_level,
             axis_y_label="CCF",
             axis_x_label="lag",
             max_lag=252*2):
    """
    Plot Cross-Correlation based on stats
 
    :x, y: Pandas series
    :adjusted: Boolean value indicating whether unbiased for CCF is True or False
    :confidence_level: confidence level informed to establish a horizontal dashed line above and below 0
 
    :return: plot
 
    """
 
    # Check if x and y type is correct
    if ((str(type(x))) != "<class 'pandas.core.series.Series'>") or (
            str(type(y)) != "<class 'pandas.core.series.Series'>"):
 
        print("Convert x into a 'pandas.core.series.Series'")
 
    else:
        backwards = ccf(x[::-1], y[::-1], unbiased=adjusted)[:max_lag]
        forwards = ccf(x, y, unbiased=adjusted)[:max_lag]
        ccf_output = np.r_[backwards[:-1], forwards]
 
        # Plotting series
        plt.figure(figsize=(12, 7), dpi=80)
        plt.bar(range(-len(ccf_output) // 2, len(ccf_output) // 2),
                ccf_output)  # Changing plot format to stem: plt.stem(range(-len(ccf_output)//2, len(ccf_output)//2), ccf_output,  use_line_collection = True)
 
        plt.xlabel(axis_x_label)
        plt.ylabel(axis_y_label)
 
        plot_title = x.name + "=f(" + y.name + ") vs " + y.name + "=f(" + x.name + ")"
 
        # xx% UCL / LCL
        plt.axhline(-confidence_level / np.sqrt(len(x)), color='k', ls='--')
        plt.axhline(confidence_level / np.sqrt(len(x)), color='k', ls='--')
 
        # Vertical Line
        plt.axvline(0, color='gray', ls='--')
 
        # Design
        plt.title(plot_title, fontsize=22)
 
    return plt.show()
 
 
def plot_multiple_series_different_frequencies(df, dash, solid, colors):
    for dsh, sol, col in zip(dash, solid, colors):
        plt.plot(df.loc[:, [dsh]], label=dsh, color=col, linestyle='dashed')
        plt.plot(df.loc[:, [sol]], label=sol, color=col, linestyle='solid')
        plt.xticks(rotation=30)
        plt.legend()
 
    plt.show()
 
############################################################################## TABLES
 
def highlight_max(s):
    if s.dtype == np.object:
        is_neg = [False for _ in range(s.shape[0])]
    else:
        is_neg = s < 0
    return ['color: red;' if cell else 'color:black'
            for cell in is_neg]
 
def generate_styled_table (df, img_name, figsave_path, dec = '{:.1f}',
                            customize = True, n_col_headers = False, n_index_name = False,
                            css_list = []):
   
    if customize:
        s = df.style.set_table_styles(css_list).format(formatter = dec, na_rep = "").apply(highlight_max)
 
    elif n_col_headers and n_index_name:
        s = df.style.set_table_styles(
                    [{'selector': 'tr:hover','props': [('background-color', 'blue'),('font-size','3em')]},
                    {'selector':'th:not(.indexname)','props':[('background-color', '#000066'),('text-align', 'center'),('color', 'white')]},
                    {'selector':'caption','props':[('text-align', 'center')]},
                    {'selector':'td','props':[('text-align', 'center')]}]
        ).format(formatter = dec , na_rep = "").apply(highlight_max)
 
 
    elif n_col_headers == False:
        s = df.style.set_table_styles(
                    [{'selector': 'tr:hover',
                    'props': [('background-color', 'blue')],
                    'font-size':'3em'},
                    {'selector':'th:not(.indexname)',
                    'props':[('background-color', '#000066'),('text-align', 'center'),('color', 'white')]},
                    {'selector':'caption',
                    'props':[('text-align', 'center')]},
                    {'selector':'td',
                    'props':[('text-align', 'center')]},
                    {'selector': 'thead',
                    'props':[('display', 'none')]}]
        ).format(formatter = dec, na_rep = "").apply(highlight_max)
 
    elif n_index_name == False:
        s = df.style.set_table_styles(
                    [{'selector': 'tr:hover','props': [('background-color', 'blue'),('font-size','3em')]},
                    {'selector':'th:not(.indexname)','props':[('background-color', '#000066'),('text-align', 'center'),('color', 'white')]},
                    {'selector':'caption','props':[('text-align', 'center')]},
                    {'selector':'td','props':[('text-align', 'center')]}]
        ).format(formatter = dec, na_rep = "").apply(highlight_max).hide_index()
 
    else:
        s = df.style.set_table_styles(
                    [{'selector': 'tr:hover',
                    'props': [('background-color', 'blue')],
                    'font-size':'3em'},
                    {'selector':'th:not(.indexname)',
                    'props':[('background-color', '#000066'),('text-align', 'center'),('color', 'white')]},
                    {'selector':'caption',
                    'props':[('text-align', 'center')]},
                    {'selector':'td',
                    'props':[('text-align', 'center')]},
                    {'selector': 'thead',
                    'props':[('display', 'none')]}]
        ).format(formatter = dec, na_rep = "").apply(highlight_max).hide_index()
 
    path_img = figsave_path + '\\' + img_name + '.png'
 
    dfi.export(s, path_img)
 
def plot_acf_pacf(series: pd.Series, lag: int):
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(series, lags=lag, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(series, lags=lag, ax=ax2)
 
    return plt.show()

Corporativo | Interno

"Esta mensagem e reservada e sua divulgacao, distribuicao, reproducao ou qualquer forma de uso e proibida e depende de previa autorizacao desta instituicao. O remetente utiliza o correio eletronico no exercicio do seu trabalho ou em razao dele, eximindo esta instituicao de qualquer responsabilidade por utilizacao indevida. Se voce recebeu esta mensagem por engano, favor elimina-la imediatamente."

"This message is reserved and its disclosure, distribution, reproduction or any other form of use is prohibited and shall depend upon previous proper authorization. The sender uses the electronic mail in the exercise of his/her work or by virtue thereof, and the institution takes no liability for its undue use. If you have received this e-mail by mistake, please delete it immediately."