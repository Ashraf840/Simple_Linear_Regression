import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('D:\ML\Salary_Data.csv')

df.head()
# xx = df.head().values

# Get all the vaolues of the columns of 'Years of Experience' & 'Salary' as numpy arrays (list-slicing using index-location)
# "df.values" will return a numpy-array. Thus it'll not contain the column label, but the values will be wrapped in a list-format, where each value will also wrapped independently in a list-format.
# Syntax:     df.iloc[startIndexofRow:endIndexofRow, startIndexofColumn:endIndexofCol]
# [NB]: column will start from the the 0th column and ends before the last column; this is depicted as below.
#   'x' = 'Matrix of Features'
#   'y' = 'Vector'
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values


# split the dataset into training & testing datasets. The ration will be 1:3 accordingly
# use the sci-kit learns 'train_test_split()' function
# [NB]:  Using the "random_state=0" in order to get the same splitting-result after the splitting, otherwise, the splitting will be done randomly
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3, random_state=0)



# import the LinearRegression class from sci-kit learn's "linear_model" class
from sklearn.linear_model import LinearRegression

# create an object of the 'LinearRegression' class which will be known as 'regressor'
# This 'Simple Linear Regression' model is the machine learning model & it learns on the training-set consists of the 'x_train' & the 'y_train' column-values
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# since the 'linear regression' models learns the corellation of x-set with the y-set from the training-set,
# now we can use the regressor/linear regression model to predict the target-values based on the independent-data.
# Using prediction, it'll create the vector of all the predicted salaries.
# the prediction will be done on the independent-variable ('x-set') using the 'regressor' (learned on the training-set), & will be stored inside a variable
y_pred = regressor.predict(x_test)



# ----- Plot the observations based on the real-data (Training Data Set) -----
# Visualize the training set result
# Make comparision of the predection using scatter plot
# Initially build a plot based on the test-set & then create a 'linear regression line' based on the training-set.
# this regression-line will contain the prediction of our previously trained model.
# Syntax:   plt.scatter("values of the X-coordinate", "values of the Y-coordinate")
#       [NB]: Plot the observation-points & the regression-line of two different colors.

# ----- Plot the observations based on the real-data -----
# Visualize the observations on 'red' color. Build on the data of accordingly the 'years of experience' & the real 'salaries'.
plt.scatter(x_train, y_train, color='red')

# ----- Draw the 'Linear Regression Line' on the plotted observed data ----- 
# Plot/draw the regression line on 'blue' color on the graph.
# Build the regression-line based on the x-coordinate ('years of experience') & the y-coordinate ('predicted salary').
# The predicted-salary will be calculated based on the training-set. Thus instead of the 'y_pred', we'll build the y-coordinate (prediction) based on the 'x_train' set. The same regressor will be used while executing this prediction.
plt.plot(x_train, regressor.predict(x_train), color='blue')

# Set the title of the plot
plt.title('Salary vs Years of experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# display the plot
plt.show()



# ----- Plot the observations based on the real-data (Test Data Set) -----
# [PURPOSE]: To visualize our ML model on the test-dataset
# Visualize the test set result
# Visualize the observations on 'red' color. Build on the data of accordingly the 'years of experience' & the real 'salaries'.
plt.scatter(x_test, y_test, color='red')

# ----- Draw the 'Linear Regression Line' on the plotted observed data ----- 
# [IMPORTANT]:  Since the regressor is trained based on the training set, we don't need to change the training-set for X-coordinate & the Y-coordinate
plt.plot(x_train, regressor.predict(x_train), color='blue')

# Set the title of the plot
plt.title('Salary vs Years of experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')

# display the plot
plt.show()

# [Moral]:  The test-set just neeed to plotted on the graph, while the test-set is not required to set on the calculcation of the 'line of regression' for drawing the line on the plotted graph


