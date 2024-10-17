import yfinance as yf
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

tesla = yf.Ticker("TSLA")
df = tesla.history(period="5y")


df = df[['Close']]
df.dropna(inplace=True)


plt.figure(figsize=(12,8))
plt.plot(df['Close'], label='Tesla close Prediction',linestyle='dotted')
plt.title('Tesla Close Price History')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend('Stock Prediction')
plt.legend()
plt.show()

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        X.append(a)
        Y.append(data[i + look_back, 0])
    return np.array(X), np.array(Y)


scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)


look_back = 60  
X, y = create_dataset(scaled_data, look_back)


X = np.reshape(X, (X.shape[0], X.shape[1], 1))


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=64, epochs=10)

predictions = model.predict(X_test)

predictions = scaler.inverse_transform(predictions.reshape(-1, 1))


y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))


rmse = np.sqrt(np.mean((predictions - y_test_unscaled) ** 2))
print(f'Root Mean Squared Error: {rmse}')

plt.figure(figsize=(12,8))
plt.plot(y_test_unscaled, color='blue', label='Actual Price')
plt.plot(predictions, color='red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend('Stock Prediction')
plt.legend()
plt.show()




