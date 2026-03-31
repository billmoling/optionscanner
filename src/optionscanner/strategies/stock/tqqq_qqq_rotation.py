"""
Backtesting script for QQQ/TQQQ Momentum Rotation Strategy.

Strategy Logic:
1.  **Risk-On** (QQQ -> TQQQ) if:
    -   QQQ > SMA200 AND QQQ > SMA10
    -   OR QQQ RSI(14) < 35 (Dip Buy)
2.  **Risk-Off** (TQQQ -> QQQ) if:
    -   QQQ < SMA10 OR QQQ < SMA200
    -   OR QQQ RSI(14) > 75 (Overbought sell)
    -   OR TQQQ drops > 15% in a single week (Hard Stop)

Requires: pip install yfinance pandas numpy matplotlib
"""

import sys
import pandas as pd
import numpy as np
import datetime
import argparse
from pathlib import Path

try:
    import yfinance as yf
except ImportError:
    print("Error: yfinance is not installed. Please run 'pip install yfinance'")
    sys.exit(1)

import matplotlib.pyplot as plt

def calculate_sma(series, period):
    return series.rolling(window=period).mean()

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # Fill NaN with 50 (neutral) or propagate
    return rsi.fillna(50)

def calculate_rsi_wilder(series, period=14):
    """
    Wilder's RSI implementation (standard for TA-Lib and many platforms).
    """
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # Use exponential moving average for Wilder's method
    ma_up = up.ewm(com=period - 1, adjust=False, min_periods=period).mean()
    ma_down = down.ewm(com=period - 1, adjust=False, min_periods=period).mean()

    rsi = ma_up / ma_down
    rsi = 100 - (100 / (1 + rsi))
    return rsi

class RotationBacktest:
    def __init__(self, start_date='2010-01-01', end_date=None, initial_capital=10000.0):
        self.start_date = start_date
        self.end_date = end_date if end_date else datetime.datetime.now().strftime('%Y-%m-%d')
        self.capital = initial_capital
        self.commission = 0.001  # 0.1% per trade
        
        # Determine 5-day rolling window for weekly drop check
        self.hard_stop_window = 5 
        self.hard_stop_threshold = -0.15 # -15%

    def fetch_data(self):
        print(f"Fetching data for QQQ and TQQQ from {self.start_date} to {self.end_date}...")
        tickers = ['QQQ', 'TQQQ']
        data = yf.download(tickers, start=self.start_date, end=self.end_date, progress=False)
        
        # yfinance returns a MultiIndex columns if >1 ticker. 
        # Structure: data['Close']['QQQ']
        
        # Handle cases where column is 'Adj Close' vs 'Close'
        if 'Adj Close' in data:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        self.qqq = prices['QQQ']
        self.tqqq = prices['TQQQ']
        
        # Align dates
        self.df = pd.DataFrame(index=self.qqq.index)
        self.df['QQQ'] = self.qqq
        self.df['TQQQ'] = self.tqqq
        self.df = self.df.dropna()
        
    def calculate_indicators(self):
        # Indicators on QQQ
        self.df['SMA200'] = calculate_sma(self.df['QQQ'], 200)
        self.df['SMA10'] = calculate_sma(self.df['QQQ'], 10)
        self.df['RSI'] = calculate_rsi_wilder(self.df['QQQ'], 14)
        
        # Helper for Hard Stop: TQQQ 5-day return
        self.df['TQQQ_5D_Ret'] = self.df['TQQQ'].pct_change(self.hard_stop_window)

    def run(self):
        # We need a loop to simulate position state
        # State: 'CASH', 'QQQ', 'TQQQ'
        # Assume we start in QQQ (Base Asset) as per instruction "Base Asset: QQQ"
        
        position = 'QQQ' 
        equity = self.capital
        shares = 0
        
        history = []
        
        # Initial Buy QQQ
        start_price = self.df['QQQ'].iloc[0]
        shares = (equity * (1 - self.commission)) / start_price
        
        print("Starting Backtest...")
        
        for i in range(len(self.df)):
            date = self.df.index[i]
            row = self.df.iloc[i]
            
            # Skip if indicators are NaN (start of data)
            if np.isnan(row['SMA200']):
                # Record equity based on current holding
                current_price = row[position]
                curr_equity = shares * current_price
                history.append({'Date': date, 'Equity': curr_equity, 'Position': position})
                continue

            # Check Hard Stop First (Safety)
            hard_stop_triggered = False
            if position == 'TQQQ':
                if row['TQQQ_5D_Ret'] < self.hard_stop_threshold:
                    # HARD STOP: Rotate back to QQQ immediately
                    # Sell TQQQ
                    sell_price = row['TQQQ']
                    proceeds = shares * sell_price * (1 - self.commission)
                    
                    # Buy QQQ
                    buy_price = row['QQQ']
                    shares = (proceeds * (1 - self.commission)) / buy_price
                    position = 'QQQ'
                    hard_stop_triggered = True
                    # print(f"{date.date()} HARD STOP triggered! TQQQ Drop {row['TQQQ_5D_Ret']:.2%}. Rotated to QQQ.")
            
            if not hard_stop_triggered:
                # Normal Logic
                
                # Rule A: Risk-On (QQQ -> TQQQ)
                if position == 'QQQ':
                    risk_on_condition = (row['QQQ'] > row['SMA200']) and (row['QQQ'] > row['SMA10'])
                    dip_buy_condition = row['RSI'] < 35
                    
                    if risk_on_condition or dip_buy_condition:
                        # Rotate to TQQQ
                        sell_price = row['QQQ']
                        proceeds = shares * sell_price * (1 - self.commission)
                        
                        buy_price = row['TQQQ']
                        shares = (proceeds * (1 - self.commission)) / buy_price
                        position = 'TQQQ'
                        # print(f"{date.date()} ROTATION: QQQ -> TQQQ. Reason: {'Dip' if dip_buy_condition else 'Trend'}")

                # Rule B: Risk-Off (TQQQ -> QQQ)
                elif position == 'TQQQ':
                    risk_off_condition = (row['QQQ'] < row['SMA10']) or (row['QQQ'] < row['SMA200'])
                    spike_sell_condition = row['RSI'] > 75
                    
                    if risk_off_condition or spike_sell_condition:
                        # Rotate to QQQ
                        sell_price = row['TQQQ']
                        proceeds = shares * sell_price * (1 - self.commission)
                        
                        buy_price = row['QQQ']
                        shares = (proceeds * (1 - self.commission)) / buy_price
                        position = 'QQQ'
                        # print(f"{date.date()} ROTATION: TQQQ -> QQQ. Reason: {'Spike' if spike_sell_condition else 'Trend Break'}")

            # Record Daily Equity
            current_price = row[position]
            curr_equity = shares * current_price
            history.append({'Date': date, 'Equity': curr_equity, 'Position': position})

        self.results = pd.DataFrame(history).set_index('Date')
        
        # Benchmark: Buy and Hold QQQ
        initial_shares_qqq = (self.capital * (1 - self.commission)) / self.df['QQQ'].iloc[0]
        self.results['Benchmark_QQQ'] = initial_shares_qqq * self.df['QQQ']
        
    def report(self):
        final_equity = self.results['Equity'].iloc[-1]
        benchmark_equity = self.results['Benchmark_QQQ'].iloc[-1]
        
        total_return = (final_equity - self.capital) / self.capital
        benchmark_return = (benchmark_equity - self.capital) / self.capital
        
        days = (self.results.index[-1] - self.results.index[0]).days
        years = days / 365.25
        cagr = (final_equity / self.capital) ** (1 / years) - 1
        benchmark_cagr = (benchmark_equity / self.capital) ** (1 / years) - 1
        
        # Max Drawdown
        rolling_max = self.results['Equity'].cummax()
        drawdown = (self.results['Equity'] - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        print("\n" + "="*40)
        print(" BACKTEST RESULTS ")
        print("="*40)
        print(f"Start Date: {self.results.index[0].date()}")
        print(f"End Date:   {self.results.index[-1].date()}")
        print(f"Duration:   {years:.2f} years")
        print("-" * 40)
        print(f"Initial Capital: ${self.capital:,.2f}")
        print(f"Final Equity:    ${final_equity:,.2f}")
        print("Benchmark (QQQ Buy&Hold): ${:,.2f}".format(benchmark_equity))
        print("-" * 40)
        print(f"Total Return:    {total_return:.2%}")
        print(f"CAGR:            {cagr:.2%}")
        print(f"Benchmark CAGR:  {benchmark_cagr:.2%}")
        print(f"Max Drawdown:    {max_dd:.2%}")
        
        # Trades
        trades = (self.results['Position'] != self.results['Position'].shift(1)).sum()
        print(f"Total Rotations: {trades}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['Equity'], label='Strategy')
        plt.plot(self.results.index, self.results['Benchmark_QQQ'], label='QQQ Buy & Hold', alpha=0.7)
        plt.title('QQQ/TQQQ Momentum Rotation vs QQQ')
        plt.yscale('log')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        output_path = 'backtest_result.png'
        plt.savefig(output_path)
        print(f"\nChart saved to {output_path}")

if __name__ == "__main__":
    backtester = RotationBacktest(start_date='2010-06-01') # TQQQ inception around 2010
    backtester.fetch_data()
    backtester.calculate_indicators()
    backtester.run()
    backtester.report()
