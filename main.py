import numpy as np
import pandas as pd

from gradient_descent import gradient_descent, model

# Training dataset
train_weight = np.array([2.9, 2.3, 0.5, 3.1, 2.5, 1.8, 2.0, 3.4, 2.2, 1.7])
train_height = np.array([3.2, 1.9, 1.4, 3.0, 2.1, 1.7, 1.8, 3.6, 2.0, 1.5])
df = pd.DataFrame({"weight": train_weight, "height": train_height})

# Testing dataset
test_weight = np.array([2.7, 2.1, 0.8])
test_height = np.array([3.1, 2.0, 1.3])
test_df = pd.DataFrame({"weight": test_weight, "height": test_height})

b, m = gradient_descent(
    df, intercept_init=-5, slope_init=2, learning_rate=0.01, epochs=500
)
print(f"y_predict: \n {model(b, m, test_df.weight)} \n y_true: \n {test_df.height}")
