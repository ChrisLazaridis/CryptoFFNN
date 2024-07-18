import pandas as pd
import mplfinance as mpf

# Load data from CSV
csv_file = 'BTCUSDT5.csv'
df = pd.read_csv(csv_file, index_col=0, parse_dates=True)

# Slice DataFrame to select only the last 30 rows
df_last_30 = df.iloc[-30:]

# Plot candlestick chart for the last 30 timestamps
mpf.plot(df_last_30, type='candle', style='charles', volume=True)
