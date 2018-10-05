import numpy as np
import random

from q2_neural import forward_backward_prop

def update_parameters(params, grad, learning_rate):
    for ix in range(0,len(params)):
        params[ix] = params[ix] - learning_rate * grad[ix]

    return params

def traininig_net():
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    n_iterations = 10000
    learning_rate = 0.01

    for i in range(0, n_iterations):
        cost, grad = forward_backward_prop(data, labels, params, dimensions)    
        params = update_parameters(params, grad, learning_rate)
        
        if i % 100 == 0:
            print("Iteration %d cost: %f"%(i, cost))

if __name__ == "__main__":
    traininig_net()