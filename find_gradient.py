import numpy as np
import matplotlib.pyplot as plt
import copy

X = np.arange(0,20,1)
X_train = np.array(X)

y_train = X_train**2 + 1


def computeCost(x, y, w, b):
    m = x.shape[0]
    totalCost = 0
    for i in range(m):
        y_hat = np.dot(w,x[i]) + b
        errorSq =  (y_hat - y[i]) ** 2
        totalCost += errorSq
        
    cost = totalCost / (2*m)
    return cost
        
def derivative(x,y,w,b):
    m = x.shape[0]
    n = x.shape[1]
    dj_dw = np.zeros((n))
    dj_db = 0
    for i in range(m):
        predicted_value = np.dot(w,x[i,:]) + b
        for j in range(n):
            err = (predicted_value - y[i]) * x[i, j]
            dj_dw[j] += err
            
        dj_db += predicted_value - y[i]
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    return dj_dw, dj_db

def gradient_descent(x,y,w_in,b_in,derivative,iteration,computecost,alpha):
    j_history = []
    w = copy.deepcopy(w_in)
    b = copy.deepcopy(b_in)
    
    for i in range(iteration):
        
        dj_dw, dj_db = derivative(x,y,w,b)
        w = w - (alpha * dj_dw)      # w is a vector
        b = b - (alpha * dj_db)
        
        j_history.append(computecost(x,y,w,b))
    
    return b,w, j_history


b_init = 0.
w_init = np.zeros(4)
iteration = 10
alpha = 9.9e-7
# b_value, w_value, cost_value = gradient_descent(X_train, y_train,w_init,b_init,derivative,iteration, computeCost,alpha)

def run_gradient_descent(x,y,iterations,alphas):
    
    n = x.shape
    w_init = np.zeros((n))
    b_init = 0.0
    
    w_val, b_val, cost_val = gradient_descent(x, y, w_init, b_init, derivative, iterations, computeCost,alphas)
    return w_val, b_val, cost_val

w_val, b_val, cost_val = run_gradient_descent(X_train,y_train,iterations=10,alphas=9.9e-7)