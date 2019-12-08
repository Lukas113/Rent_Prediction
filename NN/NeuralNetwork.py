# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 14:06:30 2019

@author: Lukas
"""

import numpy as np
import sympy as sp
import pandas as pd

class NeuralNetwork(object):
    
    def __init__(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, 
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                 max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                 beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        """initializes a NeuralNetwork object with the specified parameters"""
        #var initialization and declaration
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation = activation
        self.__solver = solver
        self.__alpha = alpha
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__learning_rate_init = learning_rate_init
        self.__power_t = power_t
        self.__max_iter = max_iter
        self.__shuffle = shuffle
        self.__random_state = random_state
        self.__tol = tol
        self.__verbose = verbose
        self.__momentum = momentum
        self.__nesterovs_momentum = nesterovs_momentum
        self.__early_stopping = early_stopping
        self.__validation_fraction = validation_fraction
        self.__beta_1 = beta_1
        self.__beta_2 = beta_2
        self.__epsilon = epsilon
        self.__n_iter_no_change = n_iter_no_change
        self.__max_fun = max_fun
        
        self.__activation_function = self.__get_activation_function()
        self.__activation_function_derivative = sp.diff(self.__activation_function)
        
    
    def fit(self, X, y):
        self.__input = X
        self.__y = y
        if self.__batch_size == 'auto':
            self.__batch_size = min(200, X.shape[0])
        #initializes weights according to `hidden_layers` (hidden + input wights +1 'bias')
        self.__weights = []
        r = np.random.RandomState(self.__random_state)
        for i in range(len(self.__hidden_layer_sizes)+1):
            if i == 0: #input weight matrix with m = num of target neurons & n = len of input vector
                self.__weights.append(r.rand(self.__hidden_layer_sizes[i], self.__input.shape[1]+1))
            elif i == len(self.__hidden_layer_sizes): #output layer weights
                self.__weights.append(r.rand(self.__y.shape[1], self.__hidden_layer_sizes[i-1]+1))
            else: #hidden layer weights
                self.__weights.append(r.rand(self.__hidden_layer_sizes[i], self.__hidden_layer_sizes[i-1]+1))
        self.__weights = np.asarray(self.__weights)
        self.__prediction = None
        for i in range(0, X.shape[0], self.__batch_size):
            minibatch = X.iloc[i:i+self.__batch_size]
            if self.__prediction is not None:
                minibatch_pred = self.__feedforward(minibatch).T
                self.__prediction = np.insert(self.__prediction, self.__prediction.shape[0], minibatch_pred, axis = 0)
            else:
                self.__prediction = self.__feedforward(minibatch).T
        return self.__prediction
    
    def predict(self, X):
        return self.__feedforward(X)
    
    def get_params(self, deep=True):
        return None
    
    def score(self, X, y, sample_weight=None):
        return None
    
    def set_params(self, **params):
        return None
    
    def __feedforward(self, batch):
        x = sp.symbols('x')
        #prepare pandas df for np.dot
        step_result = batch.to_numpy().T
        count = 1
        max_count = len(self.__weights)
        for layer_weights in self.__weights:
            #add bias value of 1 at the bottom of each vector in the matrix
            step_result = np.insert(step_result, step_result.shape[0], 1, axis = 0)
            #matrix multiplication of batch vectors & weights
            step_result = np.dot(layer_weights, step_result)
            if count < max_count:
                #perform pre-definced activation function on each value of the result matrix
                step_result = np.asarray([[self.__activation_function.subs(x, value) for value in row] for row in step_result])
            count += 1
            #print(step_result)
        return step_result
    
    def __get_activation_function(self):
        x = sp.symbols('x')
        if self.__activation == 'relu':
            return sp.Max(0, x)
        elif self.__activation == 'tanh':
            return sp.tanh(x)
        elif self.__activation == 'logistic':
            return 1 / (1 + sp.exp(-x))
        elif self.__activation == 'idendity':
            return x
        else:
            raise Exception('no valid activation function, try: "relu", "tanh", "logistic" or "idendity"')
        
    def __sum_squared_errors(self):
        return sum((self.__y.to_numpy() - self.__prediction)**2)
    
    
    
if __name__ == '__main__':
    nn = NeuralNetwork(hidden_layer_sizes = (5, 10,), random_state = 123)
    d = {'col1': [1, 2, 3, 2], 'col2': [4, 5, 6, 1]}
    X = pd.DataFrame(data=d)
    d_y = {'col_y': [9, 10, 11, 4]}
    y = pd.DataFrame(data=d_y)
    res = nn.fit(X, y)
    print(res)
    #print(nn.__sum_squared_errors())
    
    
    
    
    
    
    
    
    
    
    
    
    
    