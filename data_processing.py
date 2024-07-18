import pandas as pd
import ta


def calculate_indicators(df):
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['high'], df['low'], df['close'], df['volume'], window=14)
    df['RSI'] = ta.momentum.rsi(df['close'], window=14)
    df['MACD'] = ta.trend.macd_diff(df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['ATR'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['SMA'] = ta.trend.sma_indicator(df['close'], window=50)
    df['EMA'] = ta.trend.ema_indicator(df['close'], window=50)
    df['BB_High'] = ta.volatility.bollinger_hband(df['close'], window=20, window_dev=2)

    return df


def find_good_buys(df):
    df['future_close'] = df['close'].shift(-12)
    df['price_change'] = (df['future_close'] - df['close']) / df['close'] * 100

    df['good_buy'] = (df['price_change'] > 1).astype(int)

    df.drop(['future_close', 'price_change'], axis=1, inplace=True)

    return df


def synchronize_data(df_original, df_indicators):
    df_indicators = df_indicators.loc[:, ~df_indicators.columns.duplicated()]
    df_processed = pd.concat([df_original, df_indicators], axis=1)
    df_processed.dropna(inplace=True)

    return df_processed


def main(symbol):
    csv_file = f"{symbol}5.csv"
    df_original = pd.read_csv(csv_file, index_col=0, parse_dates=True)

    df_with_indicators = calculate_indicators(df_original)

    # Find good buys
    df_with_good_buys = find_good_buys(df_with_indicators)
    # count the number of good buys
    good_buy = df_with_good_buys['good_buy'].sum()
    print(f"Number of good buys for {symbol}: {good_buy}")
    # also print the percentage of good buys vs every buy opportunity (amount of candles)
    print(f"Percentage of good buys for {symbol}: {good_buy / len(df_with_good_buys) * 100:.2f}%")

    # Synchronize data
    df_indicators = df_with_good_buys.copy()
    df_indicators.drop(df_original.columns, axis=1, inplace=True)

    df_processed = synchronize_data(df_original, df_indicators)
    processed_csv_file = f"{symbol}5Processed.csv"
    # clean the data in the output file if any
    with open(processed_csv_file, 'w') as f:
        f.write('')
    df_processed.to_csv(processed_csv_file)
    print(f"Data saved to {processed_csv_file}")


if __name__ == "__main__":
    main()
