import pandas as pd
import concurrent.futures
import tensorflow as tf
import logging
import sys
import os
from definition import train  # Assuming train is defined in a separate file

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Allow GPU memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Disable TensorFlow warnings
tf.get_logger().setLevel(logging.ERROR)

# Set TensorFlow to use a specific GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set TensorFlow to use only the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


def load_csv(symbol):
    df = pd.read_csv(f'Normalized Data/{symbol}5Normalized.csv', index_col=0, parse_dates=True)
    return symbol, df


def main():
    try:
        # Load data for each symbol using ThreadPoolExecutor for parallelism
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT', 'BCHUSDT', 'EOSUSDT', 'XLMUSDT', 'TRXUSDT',
                   'BNBUSDT', 'LINKUSDT', 'DOTUSDT', 'FILUSDT', 'DASHUSDT', 'NEOUSDT', 'WAVESUSDT',
                   'ZRXUSDT', 'XMRUSDT', 'ETCUSDT', 'XTZUSDT', 'ALGOUSDT']
        df_dict = {}
        counter = 0

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit tasks and gather results
            future_to_symbol = {executor.submit(load_csv, symbol): symbol for symbol in symbols}
            for future in concurrent.futures.as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, df = future.result()
                    df_dict[symbol] = df
                    counter += 1
                    print(f"Loaded {counter}/{len(symbols)} symbols", end='\r', flush=True)
                except Exception as exc:
                    print(f"Exception occurred for symbol {symbol}: {exc}")

        # Train the model
        model = train(df_dict, 1000, 5)

        model.save('model.h5')
        print("Model saved to model.h5")
    except FileNotFoundError:
        print("Data file not found. Please run data_processing.py first.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Data file is empty. Please check the data file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
