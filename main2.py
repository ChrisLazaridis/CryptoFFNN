import pandas as pd
import numpy as np
import tensorflow as tf
import logging
import os
from definition import train_model
from sklearn.model_selection import train_test_split

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
    return df


def main():
    symbol = 'BTCUSDT'
    df = load_csv(symbol)

    # Use only the 'close' prices and reshape for input shape of (15,)
    close_prices = df['close'].values
    x = np.array([close_prices[i:i + 15] for i in range(len(close_prices) - 15)])
    y = close_prices[15:]

    # Split data into 80% train and 20% test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    input_shape = (5,)
    model = train_model(input_shape, x_train, y_train)

    # Save the model
    model.save('model.h5')
    print("Model saved to model.h5")

    # Evaluate the model
    test_predictions = model.predict(x_test).flatten()
    test_loss = np.mean((test_predictions - y_test) ** 2)
    print(f"Test Loss: {test_loss}")


if __name__ == "__main__":
    main()
