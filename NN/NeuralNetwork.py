# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:06:30 2019

@author: Lukas
"""

import numpy as np
import pandas as pd
import pickle
import copy

class NeuralNetwork(object):
    """
    This class provides a not yet finished feedforward regression neural network (NN)
    - hidden_layer_sizes: hidden_layer_neurons in tuple format e.g. (10,5,)
    - activation: activation function of the NN; 'logistic', 'tanh' or 'idendity'
    - batch_size: desired mini-batch size
    - random_state: random-state of weight initalisation
    
    - shuffling and epoche wise gradient descent not yet implemented
    """
    
    def __init__(self, hidden_layer_sizes=(100, ), activation='logistic', alpha=0.0001, 
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001,
                 max_iter=200, shuffle=True, random_state = 123):
        """initializes a NeuralNetwork object with the specified parameters"""
        #var initialization and declaration
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation = activation
        self.__alpha = alpha
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_rate_init = learning_rate_init
        self.__shuffle = shuffle
        self.__random_state = random_state
        self.__step_results = []
        
    def store(self, path):
        if len(path) < 4:
            path = path + '.pkl'
        elif path[-4:] != '.pkl':
            path = path + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        if len(path) < 4:
            path = path + '.pkl'
        elif path[-4:] != '.pkl':
            path = path + '.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def fit(self, X, y):
        self.__input = X
        self.__y = y.to_numpy()
        if self.__batch_size == 'auto':
            self.__batch_size = min(200, X.shape[0])
            
        #initializes weights according to `hidden_layers` (hidden + input wights +1 'bias')
        self.__weights = []
        r = np.random.RandomState(self.__random_state)
        for i in range(len(self.__hidden_layer_sizes)+1): #initialize weights randomly between -0.7 and +0.7
            if i == 0: #input weight matrix with m = num of target neurons & n = len of input vector
                self.__weights.append(r.rand(self.__hidden_layer_sizes[i], self.__input.shape[1]+1)*1.4-0.7)
            elif i == len(self.__hidden_layer_sizes): #output layer weights
                self.__weights.append(r.rand(self.__y.shape[1], self.__hidden_layer_sizes[i-1]+1)*1.4-0.7)
            else: #hidden layer weights
                self.__weights.append(r.rand(self.__hidden_layer_sizes[i], self.__hidden_layer_sizes[i-1]+1)*1.4-0.7)
        
        #perform batch-wise prop
        for i in range(0, X.shape[0], self.__batch_size):
            minibatch = X.iloc[i:i+self.__batch_size]
            tmp_res, error = NeuralNetwork.forward_prop(minibatch, self.__y, self.__weights)
            grad, self.__weights = NeuralNetwork.backprob(minibatch, self.__y, self.__weights, tmp_res, update = True)
    
    def predict(self, X):
        tmp_res, error = NeuralNetwork.forward_prop(X, None, self.__weights)
        return tmp_res[-1]
    
    @staticmethod
    def forward_prop(X, y, thetas):
        step = X.to_numpy().T
        count = 1
        max_count = len(thetas)
        tmp_res = [step]
        for theta in thetas:
            activation = True
            if count == max_count:
                activation = False
            step = NeuralNetwork.affine_forward(step, theta, activation)
            count += 1
            tmp_res.append(step)
            
        tmp_res[-1] = tmp_res[-1].T
        error = None
        if y != None:
            error = NeuralNetwork.sum_squared_errors(tmp_res[-1], y)
        return tmp_res, error
    
    @staticmethod
    def affine_forward(X, theta, activation):
        X_b = np.insert(X, X.shape[0], 1, axis = 0) #add bias input to matrix
        out = np.dot(theta, X_b)
        if activation:
            out = NeuralNetwork.log_sig(out)
        return out
    
    @staticmethod
    def log_sig(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def log_sig_der(x):
        return NeuralNetwork.log_sig(x) * (1 - NeuralNetwork.log_sig(x))
    
    @staticmethod
    def backprob(X, y, thetas, tmp_res, update):
        #C = cost_function, a = activation_function, z = scalar_product(weights*a^-1+bias)
        thetas_c = copy.deepcopy(thetas)
        gradient = NeuralNetwork.cost_backward(tmp_res[-1], y) #dC/da
        w_grad = [] #weight gradient structure
        
        for i in range(1, len(thetas)+1):
            gradient_weight = NeuralNetwork.weight_gradient(gradient, tmp_res[-i-1]) #dz/dw :D
            w_grad.insert(0, gradient_weight)
            if update:
                thetas[-i] -= gradient_weight * 0.001 #no need to change thetas in backprob testing
            gradient = NeuralNetwork.linear_layer_gradient(gradient, thetas_c[-i][:,:-1]) #dz/da^(l-1) [:,:-1] is to remove the bias from the weight layer
            gradient = NeuralNetwork.log_sig_der(gradient) #da/dz :D
            
        return w_grad, thetas
    
    @staticmethod
    def linear_layer_gradient(gradient, theta):
        """dz/da^(l-1)"""
        return gradient.dot(theta).T
    
    @staticmethod
    def weight_gradient(gradient, a_prev):
        """dz/dw -> gradient to adjust the weights"""
        a_prev_bias = np.insert(np.mean(a_prev, axis=1), a_prev.shape[0], 1, axis = 0)
        return gradient.reshape(-1,1).dot(np.array([a_prev_bias]))
    
    @staticmethod
    def cost_backward(prediction, y):
        """dC/da"""
        out = 2 * (prediction - y)
        gradient = np.mean(out, axis=0)
        return gradient
        
        
    def __activation_fun(self, x, deriv = False): #not in use yet
        if self.__activation == 'logistic':
            if deriv:
                return (1 / (1 + np.exp(-x))) * (1 - (1 / (1 + np.exp(-x))))
            else:
                return 1 / (1 + np.exp(-x))
        elif self.__activation == 'tanh':
            if deriv:
                return 1 - np.tanh(x)**2
            else:
                return np.tanh(x)
        elif self.__activation == 'idendity':
            if deriv:
                return 1
            else:
                return x
        else:
            raise Exception('no valid activation function, try: "tanh", "logistic" or "idendity"')
    
    @staticmethod
    def sum_squared_errors(prediction, y):
        return np.sum((prediction - y)**2)
    
    @staticmethod
    def euq_dist(x1, x2):
        x = x1 - x2
        return np.linalg.norm(x)
    
    @staticmethod
    def eval_grad(f, thetas, X, y, verbose = False):
        """
        - f is a function
        - x is the input of the function f (numpy array)
        check: (f(x+h) - f(x-h)) / (2*h) where h is 0.0001
        """
        h = 0.0001
        grad = copy.deepcopy(thetas)
        for i in range(len(thetas)):
            for ix in np.ndindex(thetas[i].shape):
                orig_val = thetas[i][ix]
                thetas[i][ix] = orig_val + h
                tmp_res, f_p = f(X, y, thetas) # f(x + h)
                thetas[i][ix] = orig_val - h
                tmp_res, f_m = f(X, y, thetas) # f(x - h)
                thetas[i][ix] = orig_val #restore original value
                grad[i][ix] = (f_p - f_m) / (2*h)
                if verbose:
                    print(ix, grad[i][ix])
        
        return grad
    
    @staticmethod
    def grad_check(X, y, hidden_layer_sizes = (1,)): 
        #initializes weights according to `hidden_layers` (hidden + input wights +1 'bias')
        thetas = []
        r = np.random.RandomState(123)
        for i in range(len(hidden_layer_sizes)+1): #initialize weights randomly between -0.7 and +0.7
            if i == 0: #input weight matrix with m = num of target neurons & n = len of input vector
                thetas.append(r.rand(hidden_layer_sizes[i], X.shape[1]+1)*1.4-0.7)
            elif i == len(hidden_layer_sizes): #output layer weights
                thetas.append(r.rand(y.shape[1], hidden_layer_sizes[i-1]+1)*1.4-0.7)
            else: #hidden layer weights
                thetas.append(r.rand(hidden_layer_sizes[i], hidden_layer_sizes[i-1]+1)*1.4-0.7)
        
        print('\nthetas:\n', thetas)
        y = y.to_numpy()
        tmp_res, error = NeuralNetwork.forward_prop(X, y, thetas)
        print('\ntmp_res:\n', tmp_res)
        grad, t = NeuralNetwork.backprob(X, y, thetas, tmp_res, update = False) #nothing toDo with t
        f = NeuralNetwork.forward_prop
        grad_approx = NeuralNetwork.eval_grad(f, thetas, X, y)
        print('\ngradient:\n', grad)
        print('\ngradient_approx:\n', grad_approx)
            
    
if __name__ == '__main__':
    nn = NeuralNetwork(hidden_layer_sizes = (5, 10,), random_state = 123, activation = 'logistic', batch_size = 4)
    #d = {'col1': [1., 2., 3., 2.], 'col2': [4., 5., 6., 1.]}
    d = {'col1': [2.], 'col2': [4.]}
    X = pd.DataFrame(data=d)
    #d_y = {'col_y': [9., 10., 11., 4.]}
    d_y = {'col_y': [9.]}
    y = pd.DataFrame(data=d_y)
    NeuralNetwork.grad_check(X, y, (1,))
    #nn.fit(X, y)
    #print(nn.predict(X))
    
    
    
    
    
    
    
    
    
    
    
    