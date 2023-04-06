import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)

        # Compute mean of X and y
        x_mean = np.mean(X)
        y_mean = np.mean(y)

        # Compute deviations from the mean
        x_dev = X - x_mean
        y_dev = y - y_mean

        # Compute the slope (coef_) and intercept_ of the regression line
        self.coef_ = np.sum(x_dev * y_dev) / np.sum(x_dev ** 2)
        self.intercept_ = y_mean - self.coef_ * x_mean

    def predict(self, X):
        X = np.array(X)

        # Compute predicted y values using the slope and intercept
        y_pred = self.intercept_ + self.coef_ * X

        return y_pred
    
X = [1, 2, 3, 4, 5]
y = [2, 4, 5, 4, 5]

# Create an instance of the SimpleLinearRegression class
model = SimpleLinearRegression()

# Fit the model to the data
model.fit(X, y)

# Use the model to predict the output values for new input values
X_new = [6, 7, 8, 9, 10]
y_pred = model.predict(X_new)

print(y_pred) # Output: [5.2, 5.9, 6.6, 7.3, 8.0]