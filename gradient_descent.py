import sympy as sp
import matplotlib.pyplot as plt


def model(intercept, slope, weight):
    """
    Linear regression model.

    Parameters:
    intercept (float): Intercept of the linear model.
    slope (float): Slope of the linear model.
    weight (float): Input feature (e.g., weight).

    Returns:
    float: Predicted value.
    """
    y_pred = intercept + slope * weight
    return y_pred


def rss(y_true, y_pred):
    """
    Compute the residual sum of squares (RSS).

    Parameters:
    y_true (float): True values.
    y_pred (float): Predicted values.

    Returns:
    float: RSS.
    """
    return (y_true - y_pred) ** 2


def calculate_derivative():
    """
    Calculate derivatives of the RSS with respect to intercept and slope symbolically.

    Returns:
    tuple: Tuple of derivative expressions (drss_dintercept, drss_dslope).
    """
    intercept, slope, weight, height = sp.symbols("intercept slope weight height")
    y_pred = model(intercept, slope, weight)
    rss_eq = rss(y_true=height, y_pred=y_pred)
    drss_dintercept = sp.diff(rss_eq, intercept)
    drss_dslope = sp.diff(rss_eq, slope)
    return drss_dintercept, drss_dslope


def plot_loss_function(loss_values):
    """
    Plot the loss function values over epochs.

    Parameters:
    loss_values (list): A list of loss function values for each epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(loss_values)), loss_values, marker="o", linestyle="-")
    plt.title("Loss Function Over Epochs")
    plt.xlabel("Epochs/10")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()


def gradient_descent(
    df,
    intercept_init,
    slope_init,
    learning_rate=0.1,
    epochs=1000,
    weights="weight",
    target="height",
):
    """
    Perform gradient descent for linear regression.

    Parameters:
    df (DataFrame): DataFrame containing the dataset.
    intercept_init (float): Initial value for intercept.
    slope_init (float): Initial value for slope.
    learning_rate (float): Learning rate for gradient descent.
    epochs (int): Number of epochs (iterations).
    weights (str): Column name for input feature (e.g., 'weight').
    target (str): Column name for target variable (e.g., 'height').

    Returns:
    tuple: Tuple of final intercept and slope values.
    """
    convergence_threshold = 0.00001
    previous_loss = 0
    loss_values = []
    new_intercept = intercept_init
    new_slope = slope_init
    height_df = df[target]
    weight_df = df[weights]
    d_intercept, d_slope = calculate_derivative()
    intercept, slope, weight, height = sp.symbols("intercept slope weight height")
    f_intercept = sp.lambdify((height, intercept, slope, weight), d_intercept)
    f_slope = sp.lambdify((height, intercept, slope, weight), d_slope)

    for epoch in range(epochs):
        intercept_eval = f_intercept(
            height_df, new_intercept, new_slope, weight_df
        ).sum()
        slope_eval = f_slope(height_df, new_intercept, new_slope, weight_df).sum()
        intercept_step_size = intercept_eval * learning_rate
        slope_step_size = slope_eval * learning_rate
        new_intercept -= intercept_step_size
        new_slope -= slope_step_size

        if epoch % 10 == 0 or epoch == 0:
            loss_function = sum(
                rss(df[target], model(new_intercept, new_slope, df[weights]))
            )
            loss_values.append(loss_function)

            if abs(loss_function - previous_loss) < convergence_threshold:
                plot_loss_function(loss_values)
                return new_intercept, new_slope
            else:
                previous_loss = loss_function

    plot_loss_function(loss_values)
    return new_intercept, new_slope
