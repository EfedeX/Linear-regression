# Linear Regression Gradient Descent

This Python script performs linear regression using gradient descent. It includes functions for defining the linear model, calculating derivatives symbolically, and running gradient descent with convergence monitoring and plotting.

## How to Use

1. Import the necessary libraries and functions:

   ```python
   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   import sympy as sp
   from gradient_descent import gradient_descent
2. Create a DataFrame containing your dataset with columns for "weight" and "height."

3. Define initial values for intercept and slope.
```python
b, m = gradient_descent(df, intercept_init=0, slope_init=10, learning_rate=0.01, epochs=500)
```
4. Call the gradient_descent function with your DataFrame, initial values, and desired hyperparameters.

The function will perform gradient descent and plot the loss function over epochs. It will return the final intercept and slope values.

<b>Feel free to customize the hyperparameters and initial values for your specific dataset.

Make sure to install these libraries using pip if you haven't already.