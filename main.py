import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sig(x, w, b):
    return 1 / (1 + np.exp(-(w * x + b)))

# Mean squared error function
def error(X, Y, w, b):
    err = 0
    for x, y in zip(X, Y):
        fx = sig(x, w, b)
        err += (fx - y) ** 2
    return 0.5 * err

# Gradient with respect to b
def grad_b(x, y, w, b):
    fx = sig(x, w, b)
    return (fx - y) * fx * (1 - fx)

# Gradient with respect to w
def grad_w(x, y, w, b):
    fx = sig(x, w, b)
    return (fx - y) * fx * (1 - fx) * x

# Gradient descent with stopping criterion based on error threshold
def do_gradient_descent(X, Y, w, b, eta, error_threshold):
    epoch = 0
    errors = []  # List to store the error at each epoch
    while True:
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(x, y, w, b)
            db += grad_b(x, y, w, b)
        
        # Update weights and bias
        w -= eta * dw
        b -= eta * db
        
        # Calculate the error
        err = error(X, Y, w, b)
        
        # Store the error at this epoch
        errors.append(err)
        
        # Check if error is below the threshold
        if err < error_threshold:
            print(f"Converged after {epoch+1} epochs with error: {err}")
            break
        
        epoch += 1
    
    return w, b, errors

# Data
X = [0.5, 2.5]
Y = [0.2, 0.9]

# Hyperparameters
w, b, eta = 1, 1, 1
error_threshold = .001  # Set the error threshold to stop the gradient descent

# Run gradient descent with error threshold and track errors
w_final, b_final, errors = do_gradient_descent(X, Y, w, b, eta, error_threshold)

# Output the final weights, bias, and error
print(f"Final weights: {w_final}")
print(f"Final bias: {b_final}")
print(f"Final error: {errors[-1]}")

# Plotting the error vs epoch curve
plt.plot(range(len(errors)), errors, label="Error")
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Error vs Epochs (Gradient Descent)')
plt.grid(True)
plt.legend()
plt.show()
