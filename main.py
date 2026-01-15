# ==========================================
# MODULE E: AI Applications - Individual Open Project
# Project: Gold Market Trend Analysis using LSTM
# Student Name: Vikash PR
# ==========================================

# --- 1. SETUP & LIBRARIES ---
import math
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error

# Set visual style for plots
plt.style.use('fivethirtyeight')
print("Libraries imported successfully.")

# --- 2. DATA UNDERSTANDING & PREPARATION ---

# 2.a Dataset Source: Yahoo Finance
# Ticker 'GC=F' is Gold Futures
print("Downloading Gold Futures Data...")
df = yf.download('GC=F', start='2019-01-01')

# Display raw data info
print(f"Data Shape: {df.shape}")
print(df.tail())

# 2.b Data Cleaning & Preprocessing
# We only care about the 'Close' price for trend analysis
data = df.filter(['Close'])
dataset = data.values

# Normalize the data (Scale between 0 and 1 for LSTM stability)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# 2.c Split Data into Training (80%) and Testing (20%)
training_data_len = math.ceil(len(dataset) * 0.8)
train_data = scaled_data[0:training_data_len, :]

# Create x_train and y_train data structures
# We use a 60-day window: The model learns from the past 60 days to predict the 61st day.
x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])

# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape data to be 3D for LSTM [samples, time steps, features]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(f"Training Data Shape: {x_train.shape}")


# --- 3. MODEL SYSTEM & DESIGN ---
# #### THIS IS WHERE WE DEFINE THE AI ####

model = Sequential()

# #### THIS IS THE LSTM LAYER (The Core AI Brain) ####
# Layer 1: LSTM with 50 neurons, return sequences True (to feed next LSTM layer)
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Layer 2: LSTM with 50 neurons, return sequences False (last LSTM layer)
model.add(LSTM(50, return_sequences=False))

# Layer 3: Dense Layer (25 neurons)
model.add(Dense(25))

# Layer 4: Output Layer (1 neuron for the predicted price)
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# --- 4. CORE IMPLEMENTATION (TRAINING) ---
# #### THIS IS WHERE THE AI LEARNS ####
print("Starting Model Training (This may take a minute)...")
model.fit(x_train, y_train, batch_size=1, epochs=3)
print("Model Training Complete.")


# --- 5. EVALUATION & ANALYSIS (Testing on Past Data) ---

# Create the testing dataset
test_data = scaled_data[training_data_len - 60: , :]
x_test = []
y_test = dataset[training_data_len:, :] # The actual values (unscaled)

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert to numpy array and reshape
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get predicted prices
predictions = model.predict(x_test)

# Inverse transform predictions (bring them back to normal price range)
predictions = scaler.inverse_transform(predictions)

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# --- 6. VISUALIZATION ---

# Prepare data for plotting
train = data[:training_data_len]
valid = data[training_data_len:].copy()
valid['Predictions'] = predictions

# Plot the data
plt.figure(figsize=(16,8))
plt.title('Gold Market Trend Analysis: Actual vs Predicted')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Training Data', 'Actual Price', 'AI Prediction'], loc='lower right')
plt.show()


# --- 7. PREDICTING TOMORROW'S PRICE ---
# #### THIS IS WHERE WE PREDICT THE FUTURE ####

# Get the last 60 days of closing price data
last_60_days = data[-60:].values

# Scale the data to be between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

# Create an empty list and append the past 60 days
X_test_new = []
X_test_new.append(last_60_days_scaled)

# Convert to numpy array and reshape
X_test_new = np.array(X_test_new)
X_test_new = np.reshape(X_test_new, (X_test_new.shape[0], X_test_new.shape[1], 1))

# Get the predicted scaled price
pred_price = model.predict(X_test_new)

# Undo the scaling to get the actual dollar amount
pred_price = scaler.inverse_transform(pred_price)

print(f"\n======================================")
print(f"PREDICTION FOR TOMORROW'S GOLD PRICE:")
print(f"${pred_price[0][0]:.2f}")
print(f"======================================")