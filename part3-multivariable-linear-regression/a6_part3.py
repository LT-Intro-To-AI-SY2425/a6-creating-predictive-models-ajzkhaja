import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# imports and formats the data
data = pd.read_csv("part3-multivariable-linear-regression/car_data.csv")
x = data[["miles", "age", "year"]].values
y = data["Price"].values

# split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# create linear regression model
model = LinearRegression().fit(x_train, y_train)

# Find and print the coefficients, intercept, and r squared values.
# Each should be rounded to two decimal places.
coefficients = np.around(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r2_value = round(model.score(x, y), 2)

print(f"Model's Linear Equation: y = {coefficients[0]}x1 + {coefficients[1]}x2 + {coefficients[2]}x3 + {intercept}")
print("R Squared value:", r2_value)

# Loop through the data and print out the predicted prices and the
# actual prices
print("***************")
print("Testing Results")
predictions = model.predict(x_test)
predictions = np.around(predictions, 2)

for i in range(len(x_test)):
    actual_price = y_test[i]
    predicted_price = predictions[i]
    input_values = x_test[i]
    print(f"Miles: {input_values[0]} Age: {input_values[1]} Year: {input_values[2]} Actual Price: {actual_price} Predicted Price: {predicted_price}")

# create scatter plots for each variable against the price
miles = data["miles"]
age = data["age"]
year = data["year"]
price = data["Price"]

fig, plots = plt.subplots(3)
plots[0].scatter(miles, price)
plots[0].set_xlabel("Miles")
plots[0].set_ylabel("Price")

plots[1].scatter(age, price)
plots[1].set_xlabel("Age")
plots[1].set_ylabel("Price")

plots[2].scatter(year, price)
plots[2].set_xlabel("Year")
plots[2].set_ylabel("Price")

print("Correlation between miles and price:", round(miles.corr(price), 2))
print("Correlation between age and price:", round(age.corr(price), 2))
print("Correlation between year and price:", round(year.corr(price), 2))

plt.tight_layout()
plt.show()
