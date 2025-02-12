# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Generate some synthetic data with a non-linear relationship
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature: 100 random points between 0 and 10
y = 0.5 * X**2 - 3 * X + np.random.randn(100, 1) * 5  # Quadratic relation with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial feature transformation (degree=2 for quadratic curve)
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Initialize the Ridge Regression model with a regularization parameter lambda (alpha)
ridge_reg = Ridge(alpha=1.0)

# Fit the model to the transformed training data
ridge_reg.fit(X_poly_train, y_train)

# Make predictions on the test set
y_pred = ridge_reg.predict(X_poly_test)

# Calculate Mean Squared Error (MSE) on test data
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error on Test Data: {mse:.2f}')

# Visualize the results
# Sort the data for a smooth curve plot
X_sorted = np.sort(X_test, axis=0)
X_poly_sorted = poly.transform(X_sorted)

# Plot the test data and the Ridge regression curve
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_sorted, ridge_reg.predict(X_poly_sorted), color='red', linewidth=2, label='Ridge Regression Curve')
plt.xlabel('Feature: X')
plt.ylabel('Target: y')
plt.title('Ridge Regression with Polynomial Features')
plt.legend()
plt.show()

# Coefficients and intercept
print(f'Ridge Regression Coefficients: {ridge_reg.coef_}')
print(f'Ridge Regression Intercept: {ridge_reg.intercept_}')
