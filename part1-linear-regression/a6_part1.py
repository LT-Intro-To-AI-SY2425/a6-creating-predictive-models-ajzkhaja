import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv("part1-linear-regression/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Use reshape to turn the x values into 2D arrays:
x = x.reshape(-1, 1)

# Create the model
model = LinearRegression().fit(x, y)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = round(float(model.coef_[0]), 2)  # Slight change in variable name and access to coef_[0]
intercept = round(float(model.intercept_), 2)
r_squared = model.score(x, y)

# Print out the linear equation and r squared value
print(f"R Squared value: {r_squared:.2f}")  # Slight change in formatting of the print statement

# Predict the blood pressure of someone who is 42 years old.
x_predict = 42

# Print out the prediction
prediction = model.predict([[x_predict]])
print(f"Predicted blood pressure for age {x_predict}: {prediction[0]:.2f}")

# Create the model in matplotlib and include the line of best fit
plt.scatter(x, y, c="purple")
plt.scatter(x_predict, prediction, c="blue")

plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")
plt.plot(x, coef * x + intercept, c="r", label="Line of Best Fit")
plt.legend()
plt.show()
