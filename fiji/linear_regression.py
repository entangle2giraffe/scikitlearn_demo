import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error

# Read the table
df = pd.read_csv("fiji_consum_exp.csv")

# Separate 60 percent for train dataset and rest as test dataset
msk = np.random.rand(len(df)) < 0.6
train = df[msk]
test = df[~msk]

# Import Linear Regression model
regr = linear_model.LinearRegression()

# Train dataset
# Convert List -> Array
train_x = np.asanyarray(train("Year"))
train_y = np.asanyarray(train("Value"))

# Learning
regr.fit(train_x, train_y)
print(f"Coefficient: {float(regr.coef_)}")
print(f"Intercept: {float(regr.intercept_)}")

# Plot
plt.scatter(train.Year, train.Value, color="green")
plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], color="red")
plt.x_label("Year")
plt.y_label("Government Consumption Expense (10^6 USD)")
plt.title("Fiji Government Consumption Expense 1950 - 2005")
plt.show()

# Test dataset
test_x = np.asanyarray(test("Year"))
test_y = np.asanyarray(test["Value"])
test_y_ = regr.predict(test_x)

# Accuracy of this model
print("Mean Squared Error: %.2f" % mean_squared_error(test_y_ - test_y))
print("R2 Score: %.2f" % r2_score(test_y_ - test_y))
