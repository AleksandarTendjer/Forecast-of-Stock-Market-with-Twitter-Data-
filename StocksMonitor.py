import sys
import os
import csv
import json
from googlefinance import getQuotes
import pandas as pd

import yfinance as yf

tickers_list = ['AAPL', 'IBM', 'MSFT', 'WMT','NKE']
def download_data(company):
    df_yahoo = yf.download(['MSFT'],
    start='2020-10-01',
    end='2021-05-16',
    progress=False)
    print(df_yahoo)
    # saving the DataFrame as a CSV file
    gfg_csv_data = df_yahoo.to_csv('stocks'+company+'.csv', index = True)
    print('\nCSV String:\n', gfg_csv_data)
    #print(json.dumps(getQuotes('AAPL'), indent=2) )

if __name__ == '__main__':
    if len(sys.argv)==2:
        for comp in tickers_list:
            if sys.argv[1]==comp:
                download_data(sys.argv[1])
                exit(0)
        print(" The company is not on the Dow Jones stock market list  .")
        exit(1)
    else:
        print("Incorrect number of params. Enter only a name of the company from the Dow Jones stock market .")
        exit(1) 