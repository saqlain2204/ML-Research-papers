import numpy as np


class LinearRegression:
    def __init__(self):
        self.m = 0
        self.c = 0
        self.lr = 0.1
        self.epochs = 100
    
    def fit(self, X, y):
        n = len(y)
        for _ in range(self.epochs):
            y_pred = self.m*x + self.c
            
            dm = (-2/n)*np.sum(X * (y - y_pred))
            dc = (-2/n)*np.sum(y - y_pred)
        
            self.m -= self.lr * dm
            self.c -= self.lr * dc
        
    def predict(self, X):
        return self.m * X + self.c

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        return mse

if __name__ == "__main__":
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    model = LinearRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    print("Predictions:", predictions)
    accuracy = model.evaluate(x, y)
    print("Mean Squared Error:", accuracy)
    