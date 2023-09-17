#####################
# Welcome to Cursor #
#####################

'''
Step 1: Try generating with Cmd+K or Ctrl+K on a new line. Ask for CLI-based game of TicTacToe.

Step 2: Hit Cmd+L or Ctrl+L and ask the chat what the code does. 
   - Then, try running the code

Step 3: Try highlighting all the code with your mouse, then hit Cmd+k or Ctrl+K. 
   - Instruct it to change the game in some way (e.g. add colors, add a start screen, make it 4x4 instead of 3x3)

Step 4: To try out cursor on your own projects, go to the file menu (top left) and open a folder.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Generate the data
np.random.seed(0)
weight = np.random.normal(80, 10, 100)  # weight in kg
height = weight / 44.7 + np.random.normal(0, 0.05, 100)  # height in meters
data = pd.DataFrame({'weight': weight, 'height': height})

# Step 2: Define the model and loss function
def model(weight, w, b):
    return w * weight + b

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Step 3: Define the gradient
def gradient(data, w, b):
    y_true = data['height']
    y_pred = model(data['weight'], w, b)
    dw = -2 * np.mean((y_true - y_pred) * data['weight'])
    db = -2 * np.mean(y_true - y_pred)
    return dw, db

# Step 4: Gradient descent
def gradient_descent(data, w_init, b_init, lr, epochs):
    w = w_init
    b = b_init
    for i in range(epochs):
        dw, db = gradient(data, w, b)
        w -= lr * dw
        b -= lr * db
        if i % 10 == 0:
            print(f'Epoch {i}, Loss: {loss(data["height"], model(data["weight"], w, b))}')
    return w, b

# Step 5: Run the model
w, b = gradient_descent(data, 0.0, 0.0, 0.01, 100)

# Step 6: Plot the results
plt.scatter(data['weight'], data['height'])
plt.plot(data['weight'], model(data['weight'], w, b), color='red')
plt.xlabel('Weight (kg)')
plt.ylabel('Height (m)')
plt.show()
