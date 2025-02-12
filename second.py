import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("weight-height.csv")

# Scatter plot
plt.scatter(data["Height"], data["Weight"])
plt.title("Height vs Weight")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

# Linear regression
X = data[["Height"]]
y = data["Weight"]

model = LinearRegression()
model.fit(X, y)

# Plot regression line
plt.scatter(data["Height"], data["Weight"])
plt.plot(data["Height"], model.predict(X), color="red")
plt.title("Height vs Weight with Regression Line")
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()