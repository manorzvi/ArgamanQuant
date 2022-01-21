import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from typing import List, Union
from loguru import logger


def read_yahoo(tickers: List[str], starts: List[int], ends: List[Union[None, int]]):

    for i, (ticker, start, end) in enumerate(zip(tickers, starts, ends)):
        _df = pdr.data.DataReader(ticker, data_source='yahoo', start=start, end=end)
        if _df.index[0].to_numpy() != np.datetime64(start):
            print(f'[Warning] requested start data ({np.datetime64(start)}) for {ticker} != provided start data ({_df.index[0].to_numpy()})')
        _df['Ticker'] = ticker
        _df['Date'] = _df.index
        if i == 0:
            df = _df
        else:
            df = pd.concat((df, _df), ignore_index=True)
    
    return df


def read_av_intraday(tickers: List[str], starts: List[int], ends: List[Union[None, int]], api_key: str = None):
    if api_key is None:
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    for i, (ticker, start, end) in enumerate(zip(tickers, starts, ends)):
        try:
            _df = pdr.data.DataReader(ticker, data_source='av-intraday', start=start, end=end, api_key=api_key)
        except ValueError as ve:
            err_msg = str(ve) + ' | ' + f'api_key={api_key}'
            logger.error(err_msg)
            exit(1)
        if np.datetime64(_df.index[0]) != np.datetime64(start):
            logger.warning(f'requested start data ({np.datetime64(start)}) for {ticker} != provided start data ({np.datetime64(_df.index[0])})')
        _df['Ticker'] = ticker
        _df['Time'] = _df.index.to_series().apply(lambda x: np.datetime64(x))
        if i == 0:
            df = _df
        else:
            df = pd.concat((df, _df), ignore_index=True)
    
    return df    

def calc_pct_change_per_col(df: pd.DataFrame, skip_cols: List[str] = ['Ticker', 'Date']):
    cols = df.columns.tolist()
    for skip_col in skip_cols:
        cols.remove(skip_col)
    for col in cols:
        df[col + ' Change'] = df.groupby('Ticker')[col].pct_change()
    df.dropna(inplace=True)
    return df

def calc_histogram_per_col(
    df: pd.DataFrame, bins: int = 100, density: bool = True, show: bool = True,
    skip_cols: List[str] = ['Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume', 'Ticker', 'Date']
):
    cols = df.columns.tolist()
    for skip_col in skip_cols:
        cols.remove(skip_col)
    
    tickers = df['Ticker'].unique().tolist()
    ticker_df_dict = {ticker : pd.DataFrame for ticker in tickers}
    for k in ticker_df_dict.keys():
        ticker_df_dict[k] = df[:][df['Ticker'] == k]
    
    # if show:
    #     for k, v in ticker_df_dict.items():
    #         for col in cols:
    #             fig = px.histogram(v, x=col, title=k)
    #             fig.show()
    #     print('[Error] show is deprecated in this method')
    
    ticker_hist_dict = {k: {c: {'hist': None, 'bin_edges': None} for c in cols} for k in ticker_df_dict.keys()}
    
    for k, v in ticker_df_dict.items():
        for col in cols:
            hist, bin_edges = np.histogram(v[col], bins=bins, density=density)
            ticker_hist_dict[k][col]['hist'] = hist
            ticker_hist_dict[k][col]['bin_edges'] = bin_edges

    if show:
        for col in cols:
            fig, axs = plt.subplots(1, len(ticker_df_dict), sharey=True, tight_layout=True)
            fig.suptitle(col)
            for i, (k, v) in enumerate(ticker_df_dict.items()):
                _, bin_edges = np.histogram(v[col], bins=bins, density=density)
                axs[i].hist(v[col].to_numpy(), bins=bin_edges, density=True)#bins)#, density=density)#, stacked=True)
                axs[i].set_title(k)
            plt.show()

    return ticker_hist_dict




if __name__ == '__main__':
    starts=['2000-01-01']*2
    starts=['2021-01-01']
    ends=[None]*2
    ends=[None]
    tickers = ['GOOG', 'AAPL']
    tickers = ['GOOG']

    # df = read_yahoo(tickers, starts, ends)
    df = read_av_intraday(tickers, starts, ends)
    # df = calc_pct_change_per_col(df)
    print(df)
    # ticker_hist_dict = calc_histogram_per_col(df)
    exit(0)