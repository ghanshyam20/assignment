import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the data
data = pd.read_csv("weight-height.csv")

# Step 2: Scatter plot
plt.scatter(data["Height"], data["Weight"], alpha=0.5)
plt.title("Height vs Weight")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Step 3: Choose a model (Linear Regression)
model = LinearRegression()

# Prepare the data
X = data[["Height"]]  # Independent variable (height)
y = data["Weight"]    # Dependent variable (weight)

# Step 4: Perform regression
model.fit(X, y)

# Step 5: Plot the results
plt.scatter(data["Height"], data["Weight"], alpha=0.5, label="Data")
plt.plot(data["Height"], model.predict(X), color="red", label="Regression Line")
plt.title("Height vs Weight with Regression Line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.legend()
plt.show()

# Step 6: Compute RMSE and R²
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Step 7: Assess the quality
# - Visually: The regression line fits the data well.
# - Numerically: A low RMSE and high R² (close to 1) indicate a good fit.