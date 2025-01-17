import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

'''
**********CREATE THE MODEL**********
'''

data = pd.read_csv("part2-training-testing-data/blood_pressure_data.csv")
x = data["Age"].values
y = data["Blood Pressure"].values

# Reshape the x values into 2D arrays
x = x.reshape(-1, 1)

# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values.
# Each should be a float and rounded to two decimal places.
coef = round(float(model.coef_), 2)
bias = round(float(model.intercept_), 2)
r_squared = round(model.score(xtrain, ytrain), 2)

# Print out the linear equation and r squared value:
print(f"Model's Linear Equation: y = {coef}x + {bias}")
print(f"R Squared value: {r_squared}")

'''
**********TEST THE MODEL**********
'''

# reshape the xtest data into a 2D array
xtest = xtest.reshape(-1, 1)

# get the predicted y values for the xtest values - returns an array of the results
predicted_y = model.predict(xtest)
# round the value in the np array to 2 decimal places
predicted_y = np.around(predicted_y, 2)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")
for index in range(len(xtest)):
    actual = ytest[index]
    predicted = predicted_y[index]
    x_coord = xtest[index]
    print("x value:", float(x_coord[0]), "Predicted y value:", predicted, "Actual y value:", actual)

'''
**********CREATE A VISUAL OF THE RESULTS**********
'''
plt.figure(figsize=(5, 4))
plt.scatter(xtrain, ytrain, c="purple", label="Training Data")
plt.scatter(xtest, ytest, c="blue", label="Testing Data")
plt.scatter(xtest, predicted_y, c="red", label="Predictions")
plt.xlabel("Age")
plt.ylabel("Blood Pressure")
plt.title("Blood Pressure by Age")
plt.plot(x, coef * x + bias, c="r", label="Line of Best Fit")
plt.legend()
plt.show()
