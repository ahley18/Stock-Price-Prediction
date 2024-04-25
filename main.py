import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
df = pd.read_csv('files/apple_prices.csv')

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Set 'Date' column as the index
df.set_index('Date', inplace=True)

# Resample the data on a weekly basis and calculate the mean for each week
df_weekly = df.resample('W').mean()

# Prepare features (numerical dates) and target variable (price)
X = np.arange(len(df_weekly)).reshape(-1, 1)
y = df_weekly['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)

sample_week = np.array(range(0,53)).reshape(-1, 1)

#xgboost classifier training
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=123)
model.fit(X_train, y_train)
predictions = model.predict(sample_week)
y_pred = model.predict(X_test)

# Sort the indices based on X_test and in ascending order
sorted_indices = np.argsort(X_test[:, 0])

# Sort X_train and y_train based on the sorted indices
sorted_X_train = X_train[np.argsort(X_train[:, 0])]
sorted_y_train = y_train[np.argsort(X_train[:, 0])]

# Sort X_test and y_pred based on the sorted indices
sorted_X_test = X_test[sorted_indices]
sorted_y_pred = y_pred[sorted_indices]

#plotting
fig, (ax1,ax2) = plt.subplots(ncols = 2, figsize=(12, 6), sharex=True, sharey=True)

#APPL Weekly Price(Actual vs Predicted)
pred_price = []

for date, price in zip(sample_week, predictions):
    print(f"Week {date}, Predicted Average Price: {price:.2f}")
    pred_price.append(price)

print(sorted_X_test.shape, sorted_y_pred.shape)
print(sample_week.shape, predictions.shape)

ax1.plot(X, y, c = 'blue', label = 'Actual Price per Week')
ax1.plot(sample_week, pred_price, c = 'red', label = 'Predicted Price per Week')
ax1.set_title("APPL Weekly Price(Actual vs Predicted)")
ax1.legend()
ax1.set_xlabel('Week')
ax1.set_ylabel('Price')
ax1.grid(True)

#APPL Weekly Price(Training vs Predicted)
pred_price = []

for date, price in zip(sorted_X_test, sorted_y_pred):
    print(f"Week {date}, Predicted Average Price: {price:.2f}")
    pred_price.append(price)

print(sorted_X_test.shape, sorted_y_pred.shape)
print(sample_week.shape, predictions.shape)

ax2.plot(sorted_X_train, sorted_y_train, c = 'blue', label = 'Training Data Price per Week')
ax2.plot(sorted_X_test, sorted_y_pred, c = 'red', label = 'Predicted Price per Week')
ax2.set_title("APPL Weekly Price(Training vs Predicted)")
ax2.legend()
ax2.grid(True)
ax2.set_xlabel('Week')
ax2.set_ylabel('Price')
fig.savefig(r'files\APPL Predictions.png')
plt.show()