import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Read in the data
Data = pd.read_csv("C:\\Users\\shark\\Downloads\\archive (3)\\LinearRegTrain.csv")
testData = pd.read_csv("c:\\Users\\shark\\Downloads\\archive (3)\\LinearRegTest.csv")

#Select dependent and independent variables
x = Data['x'].values
y = Data['y'].values
testY = testData['y'].values
testX = testData['x'].values

#Visualise the raw training and testing data
trainVisualisation = Data.plot(x = 'x', y = 'y', kind = 'scatter', title = 'Training Data')
testVisualisation = testData.plot(x = 'x', y = 'y', kind = 'scatter', title = "Testing Data")
plt.show(trainVisualisation)
plt.show(testVisualisation)

#Train/fit the model on the training data
def LinearRegression(x, y, xValue, b0, b1): #Fit the model on x and y values of test set, with beta-coefficients and independent variable inputs
    yPredicted = b0 + (b1 * xValue) #Linear regression equation implementation
    print(yPredicted)
    return(yPredicted)

#Optimise the model with gradient descent
def GradientDescent(x, y, learningRate = 0.000001, iterations = 3000): #Fit gradient descent to training data and set the learning rate and n iterations
    b0 = 0 #Set the initial coefficient values to zero
    b1 = 0
    costs = [] #Initialise a list to store the costs calculated by each iteration
    for i in range(iterations):
        yPredicted = b0 + (b1 * x) #Predict y values with the current coefficient values

        b0Gradient = (-2 / 700) * np.sum(y - yPredicted) #Define the intercept
        b1Gradient = (-2 / 700) * np.sum((y - yPredicted) * x) #Define the slope

        b0 = b0 - learningRate * b0Gradient #Update the coefficients such that they take larger steps when there's a steeper gradient
        b1 = b1 - learningRate * b1Gradient

        cost = np.mean((y - yPredicted)**2) #Define the cost function (mean squared error) and apply it to the current iteration
        costs.append(cost) 

        print("cost:", cost, "b0:", b0, "b1:", b1) #Document the cost and current coefficient values
    return(b0, b1, costs)

b0, b1, costs = GradientDescent(x, y) #Run gradient descent with the beta coefficients for the regression

#Visualise the cost function
plt.plot(costs) 
plt.xlabel("Iteration")
plt.ylabel("Cost (MSE)")
plt.title("Cost by Iteration")
plt.show()

#Predict all values for the test set
regressionValues = LinearRegression(x, testY, testX, b0, b1)
testData['regression'] = regressionValues

#Visualise the result
plt.scatter(testData['x'], testData['y'], alpha = 0.6)
plt.plot(testData['x'], testData['regression'], color = 'red', label = 'Regression Line', linewidth = 2)
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Result - Scatter Plot with Regression Line')
plt.show()

check = input("Do you want to predict a value?: ")

#User interface allowing for custom inputs
while check == "Yes":
    userInput = int(input("Enter an x value to predict its y value: "))
    LinearRegression(x, testY, userInput, b0, b1)
    check = input("Do you want to predict a value?: ")
    if check != "Yes":
        break




    

    
