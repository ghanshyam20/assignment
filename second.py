import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("weight-height.csv")
X = df[["Height"]].values  
y = df["Weight"].values  

# Scatter plot
plt.scatter(X, y, alpha=0.3)
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Scatter Plot: Height vs. Weight")
plt.show()

# Linear Regression
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Plot Regression Line
plt.scatter(X, y, alpha=0.3, label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Linear Regression: Height vs. Weight")
plt.legend()
plt.show()

# Compute RMSE and R² Score
rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")
