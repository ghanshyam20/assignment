import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("diamonds.csv")
print(df.head())

# Define features and target variable
X = df[['carat', 'depth', 'table', 'x', 'y', 'z']]
y = df[['price']]
print(X, y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Define alpha values
alphas = [0.1, 0.2, 0.3, 0.4, 0.5,1,2, 3, 4, 5, 6, 7, 8]

# Initialize scores list
scores = []

# Loop through alpha values
for alp in alphas:
    lasso = linear_model.Lasso(alpha=alp)  # Fixed incorrect function name
    lasso.fit(X_train, y_train)
    print(lasso.coef_.round(2), lasso.intercept_)
    
    sc = lasso.score(X_test, y_test)
    scores.append(sc)
    
    print(f"alpha = {alp}, lasso score: {sc}")

# Plot the results
plt.plot(alphas, scores)
plt.xlabel("Alpha")
plt.ylabel("R² Score")
plt.title("Lasso Regression: Alpha vs R² Score")
plt.show()

# Find the best R² score and corresponding alpha
best_r2 = max(scores)
idx = scores.index(best_r2)
best_alp = alphas[idx]

print(f"Best R² = {best_r2}, Best alpha = {best_alp}")
