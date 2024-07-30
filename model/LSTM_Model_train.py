import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

def train_lstm_model(symbol_name, timeframe):
    # Symbol and folder name
    symbol_timeframe = f'{symbol_name}_{timeframe}'  # Example: 'NEPSE_1D', 'NEPSE_1M', etc.
    folder_name = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_data", symbol_name, timeframe)
    os.makedirs(folder_name, exist_ok=True)

    # Load the dataset
    csv_path = os.path.join(folder_name, f"{symbol_timeframe}.csv")
    df = pd.read_csv(csv_path).dropna()
    df.columns = df.columns.str.capitalize()

    df = df.set_index('Date')
    data_close = df.filter(['Close'])
    dataset = data_close.values
    training_data_len = math.ceil(len(dataset) * .8)

    # Adjust prediction lengths based on timeframe
    if timeframe == '1D':
        prediction_length = 7
    else:
        prediction_length = 4 * 60

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Prepare the training data
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    initial_window_size = 10
    for i in range(initial_window_size, len(train_data)):
        x_train.append(train_data[i - initial_window_size:i, 0])
        y_train.append(train_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], initial_window_size, 1))

    # Prepare the testing data
    test_data = scaled_data[training_data_len - initial_window_size:, :]
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(initial_window_size, len(test_data)):
        x_test.append(test_data[i - initial_window_size:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], initial_window_size, 1))

    # Build the model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(initial_window_size, 1)))
    model.add(tf.keras.layers.LSTM(100, return_sequences=True))
    model.add(tf.keras.layers.LSTM(100))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dense(1))

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Define callbacks for early stopping and model checkpoint
    early_stop = EarlyStopping(monitor='val_loss', patience=10)
    model_checkpoint = ModelCheckpoint(os.path.join(folder_name, f"{symbol_name}_model.keras"), save_best_only=True)

    # Train the model
    history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_split=0.2, callbacks=[early_stop, model_checkpoint])

    # Save the final model
    model.save(os.path.join(folder_name, f"{symbol_name}_model.keras"))

    # Save the scaler and data
    joblib.dump(scaler, os.path.join(folder_name, f"{symbol_name}_scaler.pkl"))
    np.save(os.path.join(folder_name, f"{symbol_name}_x_train.npy"), x_train)
    np.save(os.path.join(folder_name, f"{symbol_name}_y_train.npy"), y_train)
    np.save(os.path.join(folder_name, f"{symbol_name}_x_test.npy"), x_test)
    np.save(os.path.join(folder_name, f"{symbol_name}_y_test.npy"), y_test)

    # Print final training and validation loss
    print(f"Training completed and model saved. Final training loss: {history.history['loss'][-1]:.4f}, Validation loss: {history.history['val_loss'][-1]:.4f}")

train_lstm_model('NEPSE', '1D')
