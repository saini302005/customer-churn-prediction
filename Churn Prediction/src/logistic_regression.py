import numpy as np

class LogisticRegression:
    """
    Custom Logistic Regression implementation using only NumPy.
    Supports L2 regularization, convergence tolerance, and loss history tracking.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000, regularization='l2', lambda_=0.01):
        """
        Initialize the Logistic Regression model.
        
        Args:
            learning_rate (float): Step size for gradient descent.
            n_iterations (int): Maximum number of iterations.
            regularization (str): Type of regularization ('l2' only).
            lambda_ (float): Regularization strength.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_ = lambda_
        self.weights = None
        self.bias = None
        self.loss_history = []

    def sigmoid(self, z):
        """
        Compute the sigmoid function.
        σ(z) = 1 / (1 + e^(-z))
        """
        # Ensure z is a numpy array for stability
        z = np.asarray(z, dtype=float)
        # clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss with epsilon for stability.
        Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)] + regularization
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        m = len(y_true)
        
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        if self.regularization == 'l2':
            l2_loss = (self.lambda_ / (2 * m)) * np.sum(np.square(self.weights))
            loss += l2_loss
            
        return loss

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        
        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        m, n = X.shape
        # Initialize weights and bias
        self.weights = np.zeros(n)
        self.bias = 0.0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Print progress every 100 iterations
            if i % 100 == 0:
                print(f"Iteration {i}, Loss: {loss:.6f}")
            
            # Check convergence
            if i > 0 and abs(self.loss_history[-2] - self.loss_history[-1]) < 1e-6:
                print(f"Converged at iteration {i}, Loss: {loss:.6f}")
                break
            
            # Backward pass (Gradients)
            dw = (1 / m) * np.dot(X.T, (y_pred - y))
            db = (1 / m) * np.sum(y_pred - y)
            
            # Apply L2 regularization to weights gradient
            if self.regularization == 'l2':
                dw += (self.lambda_ / m) * self.weights
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Return probability scores for each sample.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        """
        Return binary predictions based on a threshold.
        """
        y_pred_proba = self.predict_proba(X)
        return (y_pred_proba >= threshold).astype(int)
