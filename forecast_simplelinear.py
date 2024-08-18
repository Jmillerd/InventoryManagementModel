import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Create or load your real sales data into a DataFrame (replace this with your data loading code)
df = pd.read_csv('data/sales_data.csv')  # Load data from a CSV file

# Ensure that 'Date' column is in datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data, test_data = train_test_split(df, train_size=train_size, shuffle=False)

# Feature engineering: You can add more features like seasonality, holidays, etc.
train_data['Day'] = train_data['Date'].dt.dayofyear
test_data['Day'] = test_data['Date'].dt.dayofyear  # Add this line for the test set

# Prepare the input features and target variable
X_train = train_data[['Day']]
y_train = train_data['Sales']
X_test = test_data[['Day']]
y_test = test_data['Sales']

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate forecast accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calculate RMSE directly
r2 = r2_score(y_test, y_pred)

# Calculate forecast bias
bias = np.mean(y_pred - y_test)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")
print(f"Forecast Bias: {bias:.2f}")

