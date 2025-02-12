import pandas as pd

# Load CSV file
df = pd.read_csv("weight-height.csv")  # Ensure 'data.csv' is in the same directory

# Display first few rows
print(df.head())

# Calculate correlation between "Experience" and "Salary"
correlation = df['Weight'].corr(df['Height'])

# Print correlation value
print("Correlation between Experience and Salary:", correlation)
 