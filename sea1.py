# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Step 1: Load the CSV file into a DataFrame
# Assuming the CSV file has columns 'Height' and 'Weight'
file_path = 're.csv'  # Replace with the actual file path
data = pd.read_csv(file_path)

# Step 2: Extract Features and Target variable
X = data['Height'].values.reshape(-1, 1)  # Feature: Height (reshape to 2D)
y = data['Weight'].values  # Target: Weight

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Apply Polynomial Feature Transformation (degree=2)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Step 5: Initialize and fit Ridge Regression model
ridge_reg = Ridge(alpha=1.0)  # Regularization parameter (alpha)
ridge_reg.fit(X_poly_train, y_train)

# Step 6: Make predictions on the test set
y_pred = ridge_reg.predict(X_poly_test)

# Step 7: Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse:.2f}')

# Step 8: Visualize the results
# Sort the data for a smooth curve plot
X_sorted = np.sort(X_test, axis=0)
X_poly_sorted = poly.transform(X_sorted)

# Plot the test data and the Ridge regression cur
