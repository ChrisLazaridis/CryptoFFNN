import tensorflow as tf
from scipy.optimize import least_squares
import numpy as np
import os

# Enable mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
# Allow GPU memory growth
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)  # Final output layer with a single neuron for regression
    ])
    return model


def model_predict(weights, input_shape, x):
    model = create_model(input_shape)
    offset = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights_shape = layer.get_weights()[0].shape
            biases_shape = layer.get_weights()[1].shape
            layer_weights = weights[offset:offset + np.prod(weights_shape)].reshape(weights_shape)
            offset += np.prod(weights_shape)
            layer_biases = weights[offset:offset + np.prod(biases_shape)].reshape(biases_shape)
            offset += np.prod(biases_shape)
            layer.set_weights([layer_weights, layer_biases])
    return model.predict(x).flatten()


iteration_counter = 0


def objective_function(weights, input_shape, x, y):
    global iteration_counter
    iteration_counter += 1
    print(f"Iteration: {iteration_counter}")
    predictions = model_predict(weights, input_shape, x)
    return predictions - y


def train_model(input_shape, x_train, y_train):
    global iteration_counter
    iteration_counter = 0  # Reset iteration counter
    model = create_model(input_shape)
    initial_weights = []
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_weights, layer_biases = layer.get_weights()
            initial_weights.extend(layer_weights.flatten())
            initial_weights.extend(layer_biases.flatten())
    initial_weights = np.array(initial_weights)

    # Define more appropriate stopping criteria
    result = least_squares(objective_function, initial_weights, args=(input_shape, x_train, y_train),
                           method='lm', xtol=1e-4, ftol=1e-4, gtol=1e-4)

    optimized_weights = result.x
    model = create_model(input_shape)
    offset = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            weights_shape = layer.get_weights()[0].shape
            biases_shape = layer.get_weights()[1].shape
            layer_weights = optimized_weights[offset:offset + np.prod(weights_shape)].reshape(weights_shape)
            offset += np.prod(weights_shape)
            layer_biases = optimized_weights[offset:offset + np.prod(biases_shape)].reshape(biases_shape)
            offset += np.prod(biases_shape)
            layer.set_weights([layer_weights, layer_biases])
    return model

# def train(df_dict, batch_size, epochs):
#     x_test_all = []
#     y_test_all = []
#     total_batches = sum([len(df) // batch_size for df in df_dict.values()]) * epochs
#
#     for symbol, df in df_dict.items():
#         x = df.drop(columns=['good_buy'])
#         y = df['good_buy']
#         x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#         x_test_all.append(x_test)
#         y_test_all.append(y_test)
#
#     model = create_model((batch_size, 12))
#     model.summary()
#
#     overall_progress = 0
#     overall_total = total_batches
#     overall_start_time = datetime.now()
#
#     not_used_symbols = []
#     for epoch in range(epochs):
#         epoch_loss = 0
#         epoch_accuracy = 0
#
#         print(f"\nEpoch {epoch + 1}/{epochs}", flush=True)
#
#         symbols = list(df_dict.keys())
#         random.shuffle(symbols)
#
#         if len(not_used_symbols) == len(symbols):
#             not_used_symbols = []
#         symbol = random.choice([s for s in symbols if s not in not_used_symbols])
#         not_used_symbols.append(symbol)
#         df = df_dict[symbol]
#         x_train = df.drop(columns=['good_buy'])
#         y_train = df['good_buy']
#
#         epoch_batches = len(x_train) // batch_size
#
#         for i in range(0, len(x_train) - batch_size, batch_size):
#             x_batch = x_train.iloc[i:i + batch_size].values.reshape(1, batch_size, 12)
#             y_batch = np.array([y_train.iloc[i]])
#             history = model.train_on_batch(x_batch, y_batch)
#             overall_progress += 1
#
#             # Update statistics
#             epoch_loss += history[0]
#             epoch_accuracy += history[1]
#
#             # Calculate progress
#             progress_percent = overall_progress / overall_total * 100
#             overall_eta_seconds = ((datetime.now() - overall_start_time) / overall_progress) * (
#                     overall_total - overall_progress)
#             overall_eta_seconds = overall_eta_seconds.total_seconds()
#
#             # Convert overall ETA to hours, minutes, and seconds
#             hours = int(overall_eta_seconds // 3600)
#             minutes = int((overall_eta_seconds % 3600) // 60)
#             seconds = int(overall_eta_seconds % 60)
#             overall_eta_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"
#             # Print progress
#             print(
#                 f"\r - Epoch Progress: {i // batch_size + 1}/{epoch_batches} - Loss: {history[0]:4f}, Accuracy: {history[1]:4f} - Overall Progress: {progress_percent: 2f}% - ETA: {overall_eta_formatted}",
#                 end='', flush=True)
#
#         # Calculate epoch statistics
#         epoch_loss /= epoch_batches
#         epoch_accuracy /= epoch_batches
#
#         print(f"\rEpoch {epoch + 1}/{epochs} - Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_accuracy:.4f}",
#               flush=True)
#
#     for symbol, x_test, y_test in zip(df_dict.keys(), x_test_all, y_test_all):
#         test_loss, test_accuracy = model.evaluate(x_test.values.reshape(-1, batch_size, 12), y_test)
#         print(f"Test Loss for {symbol}: {test_loss}, Test Accuracy: {test_accuracy}")
#
#     return model
