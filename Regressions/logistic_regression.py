import numpy as np


class LogisticRegression:
    def __init__(self):
        self.w = 0
        self.c = 0
        self.lr = 0.1
        self.epochs = 100
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n = len(y)
        
        for _ in range(self.epochs):
            y_pred = self.w*X + self.c
            y_pred = self._sigmoid(y_pred)
            
            dw = (1/n)*np.sum(X * (y_pred - y))
            dc = (1/n)*np.sum(y_pred - y)
            
            self.w -= self.lr * dw
            self.c -= self.lr * dc
        
    def predict(self, X):
        y_pred = self.w*X + self.c
        y_pred = self._sigmoid(y_pred)
        
        return np.where(y_pred >= 0.5, 1, 0)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
if __name__ == "__main__":
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 0, 0, 1, 1, 1])
    model = LogisticRegression()
    model.fit(x, y)
    predictions = model.predict(x)
    print("Predictions:", predictions)
    accuracy = model.evaluate(x, y)
    print("Accuracy:", accuracy)