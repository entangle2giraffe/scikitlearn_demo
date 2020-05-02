import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
import sys

# Import the local module
# The module import pandas and slicing dataset
# for specific country of the input of variable 'coa'
from util import country, des, instance

# Read the table
coa = input("Country:")

# Store the variable in cdf
cdf = country(coa)
# Separate 80% for train dataset and rest as test dataset
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# Import Linear Regression model
regr = LinearRegression()

# Prompt the user for degree
degree = input("Degree(only int):")
deg_int = int(degree)

# Train dataset
# Convert List -> Array
train_x = np.asanyarray(train[["Year"]])
train_y = np.asanyarray(train[["Value"]])

# Test dataset
test_x = np.asanyarray(test[["Year"]])
test_y = np.asanyarray(test[["Value"]])

# Transform x
# Polynomial
poly = PolynomialFeatures(deg_int)
train_x_poly = poly.fit_transform(train_x)
# Learning
train_y_ = regr.fit(train_x_poly, train_y)
des(coa, deg_int)
print(f"Coefficient: ", regr.coef_)
print(f"Intercept: ", regr.intercept_)

# Initialize x dimension
XX = np.arange(1950, 2011, 60/7)


def f(deg, arr):
    const = regr.intercept_[0]

    if deg <= 0:
        return 0
    else:
        yy = regr.coef_[0][deg] * np.power(arr, deg) + f(deg - 1, arr)
        return yy + const


func = f(deg_int, XX)
plt.plot(XX, func, '-r')
plt.xlabel("Year")
plt.ylabel("Value")
plt.show()

test_x_poly = poly.fit_transform(test_x)
test_y_ = regr.predict(test_x_poly)

# Plot
plt.show()

# Accuracy
print("MAE: ", mean_absolute_error(test_y_, test_y))
print("MSE: ", mean_squared_error(test_y_, test_y))
print("R2:  ", r2_score(test_y, test_y))
