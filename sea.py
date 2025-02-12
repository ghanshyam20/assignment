# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Import dataset and necessary modules from scikit-learn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------------
# 1. Load the Diabetes Dataset
# -------------------------------
diabetes = load_diabetes()

# Create a DataFrame with the features and target
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# Display the first few rows of the dataset
print("First 5 rows of the diabetes dataset:")
print(df.head())

# --------------------------------------
# 2. Visualize the Correlation Matrix
# --------------------------------------
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()

# Create a heat map to visualize correlations
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap of Diabetes Dataset')
plt.show()

# ---------------------------------------------
# 3. Split the Data and Train a Regression Model
# ---------------------------------------------
# Separate features (X) and target variable (y)
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------------------------------------
# 4. Evaluate the Model
# ---------------------------------------------
# Predict on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R^2 Score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

# ---------------------------------------------
# 5. Plot Actual vs. Predicted Values
# ---------------------------------------------
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)
plt.xlabel('Actual Diabetes Progression')
plt.ylabel('Predicted Diabetes Progression')
plt.title('Actual vs Predicted Values')
plt.show()
