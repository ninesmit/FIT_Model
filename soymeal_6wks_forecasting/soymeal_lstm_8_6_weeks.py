## ------------------------------------------------PreProcessing----------------------------------------------------------##

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Split the dataset
train_start, train_end = '2014-01-01', '2022-12-31'
val_start, val_end = '2023-01-01', '2023-12-31'

training = df_combined_filtered.loc[train_start:train_end]
validation = df_combined_filtered.loc[val_start:val_end]
test = df_combined_filtered.loc[val_end:]
# test = df_combined_filtered.loc[val_start:val_end]

# Define features to use
features = features_combined
target = ['[SM1 Comdty]_[PX_LAST]']

# Normalize the data
scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

train_features_scaled = scaler.fit_transform(training[features])
val_features_scaled = scaler.transform(validation[features])
test_features_scaled = scaler.transform(test[features])

train_target_scaled = target_scaler.fit_transform(training[['[SM1 Comdty]_[PX_LAST]']])
val_target_scaled = target_scaler.transform(validation[['[SM1 Comdty]_[PX_LAST]']])
test_target_scaled = target_scaler.transform(test[['[SM1 Comdty]_[PX_LAST]']])

# Convert time series into supervised learning format
def create_sequences(features, target, seq_length, forecast_horizon):
    x, y = [], []
    for i in range(len(features) - seq_length - forecast_horizon):
        x.append(features[i:i+seq_length])
        y.append(target[i+seq_length:i+seq_length+forecast_horizon])
    return np.array(x), np.array(y)

seq_length = 50
forecast_horizon = 30

# Create a sequence for training, validation, and test set
x_train, y_train = create_sequences(train_features_scaled, train_target_scaled, seq_length, forecast_horizon)
x_val, y_val = create_sequences(val_features_scaled, val_target_scaled, seq_length, forecast_horizon)
x_test, y_test = create_sequences(test_features_scaled, test_target_scaled, seq_length, forecast_horizon)

# Reshape for RNN input (sample, timesteps, features)
x_train = x_train.reshape(x_train.shape[0], seq_length, len(features))
x_val = x_val.reshape(x_val.shape[0], seq_length, len(features))
x_test = x_test.reshape(x_test.shape[0], seq_length, len(features))

## ---------------------------------------------------Training------------------------------------------------------------##

# Define callback fundtion
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6)

# Building a model
model = Sequential([
    LSTM(30, activation='tanh', return_sequences=True, input_shape=(seq_length, len(features))),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(30, activation='tanh', return_sequences=True, recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.2),
    LSTM(30, activation='tanh', return_sequences=False, recurrent_regularizer=l2(0.0005), bias_regularizer=l2(0.0005)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(50, activation='relu', kernel_regularizer=l2(0.0005)),
    Dropout(0.2),
    Dense(forecast_horizon)
])

model.summary()

# Compile and train the model
model.compile(optimizer=Adam(learning_rate=0.0005), loss='huber', metrics=['mae'])
history = model.fit(
    x_train, y_train, 
    validation_data=(x_val, y_val),
    epochs=70,
    batch_size=32,
    callbacks=[early_stopping, lr_scheduler]
)   


## ------------------------------------------------Validating the model---------------------------------------------------##

# Make a prediciton
y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)
y_val_pred = model.predict(x_val)

# # Convert the scale back to normal
y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1,forecast_horizon))
y_test_pred_inv = target_scaler.inverse_transform(y_test_pred)

y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1,forecast_horizon))
y_train_pred_inv = target_scaler.inverse_transform(y_train_pred)

y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1,forecast_horizon))
y_val_pred_inv = target_scaler.inverse_transform(y_val_pred)

# Calculate MAE
mae_lstm = np.abs(y_train_inv - y_train_pred_inv).mean()
print('MAE of LSTM model (Train set):', mae_lstm)
mae_lstm = np.abs(y_val_inv - y_val_pred_inv).mean()
print('MAE of LSTM model (Validation set):', mae_lstm)
mae_lstm = np.abs(y_test_inv - y_test_pred_inv).mean()
print('MAE of LSTM model (Test set):', mae_lstm)

## ------------------------------------------------Visualising-----------------------------------------------------------##

# Plot training history
plt.figure(figsize=(12,4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training Loss vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
