import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Load Data
df = pd.read_csv('flight_data_live.csv')

# Handle missing values in anomaly column (if any)
# As we saw there are some NaNs in anomaly, let's forward fill or fill with 0
df['anomaly'] = df['anomaly'].fillna(0)
# Also fill NaNs in features
df = df.fillna(0)

# Features and target
feature_cols = ['vib_x', 'vib_y', 'vib_z', 'm1', 'm2', 'm3', 'm4', 'ekf_vel_var', 'ekf_pos_var', 'ekf_comp_var']
target_col = 'anomaly'

# We want to predict anomaly at time t + horizon
# Let's say horizon is 5 steps ahead (so we predict failure before it happens)
horizon = 5
df['target'] = df[target_col].shift(-horizon)

# Drop rows with NaN targets (the last 'horizon' rows)
df = df.dropna(subset=['target'])

data = df[feature_cols].values
targets = df['target'].values

# Scale features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
sequence_length = 10
X = []
y = []

for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i : i + sequence_length])
    y.append(targets[i + sequence_length - 1])

X = np.array(X)
y = np.array(y)

print(f"Data shape: X={X.shape}, y={y.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build LSTM model
model = Sequential([
    LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Save the model
model.save('motor_failure_lstm.h5')
print("Model saved to motor_failure_lstm.h5")
