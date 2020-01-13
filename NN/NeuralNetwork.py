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
    
    def __init__(self, hidden_layer_sizes=(100, ), activation='logistic', alpha=0.0001, 
                 batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, 
                 max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, 
                 momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, 
                 beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000):
        """initializes a NeuralNetwork object with the specified parameters"""
        #var initialization and declaration
        self.__hidden_layer_sizes = hidden_layer_sizes
        self.__activation = activation
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
        self.__y = y
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
        self.__weights = np.asarray(self.__weights)
        
        #perform batch-wise forward-prop
        for i in range(0, X.shape[0], self.__batch_size):
            minibatch = X.iloc[i:i+self.__batch_size]
            self.__prediction = self.__feedforward(minibatch).T
            self.__backprop(minibatch)
    
    def predict(self, X):
        return self.__feedforward(X)[-1]
    
    def get_params(self, deep=True):
        return None
    
    def score(self, X, y, sample_weight=None):
        return None
    
    
    def __feedforward(self, batch):
        step_result = batch.to_numpy().T
        count = 1
        max_count = len(self.__weights)
        self.__step_results = [step_result]
        for layer_weights in self.__weights:
            activation = True
            if count == max_count:
                activation = False
            step_result = self.__affine_forward(step_result, layer_weights, activation)
            count += 1
            self.__step_results.append(step_result)
        return step_result
    
    def __affine_forward(self, X, theta, activation = True):
        X_bias = np.insert(X, X.shape[0], 1, axis = 0) #add bias input to matrix
        out = np.dot(theta, X_bias)
        if activation:
            out = self.__activation_fun(out)
        return out
    
    def __backprop(self, batch):
        #C = cost_function, a = activation_function, z = scalar_product(weights*a^-1+bias)
        weights = copy.deepcopy(self.__weights)
        gradient = self.__cost_backward() #dC/da
        w_grad = [] #weight gradient structure
        
        for weight_layer_index in range(1, len(self.__weights)+1):
            gradient_weight = self.__weight_gradient(gradient, self.__step_results[-weight_layer_index-1]) #dz/dw
            print('\ngradient_w:\n', gradient_weight)
            w_grad = np.concatenate((w_grad, list(reversed(gradient_weight.ravel()))), axis = 0) #concat gradient unrolled
            self.__weights[-weight_layer_index] -= gradient_weight * self.__learning_rate_init #adjust the weights with the negative gradient
            gradient = self.__linear_layer_gradient(gradient, weights[-weight_layer_index][:,:-1]) #dz/da^(l-1) [:,:-1] is to remove the bias from the weight layer
            gradient = self.__activation_fun(gradient, deriv = True) #da/dz
            
        #perform gradient eval
        w_grad = list(reversed(w_grad))
        print('\ngradient:\n', w_grad)
        #grad_approx = self.eval_grad()
            
    def __linear_layer_gradient(self, gradient, theta):
        """dz/da^(l-1)"""
        return gradient.dot(theta).T

    def __weight_gradient(self, gradient, a_prev):
        """dz/dw -> gradient to adjust the weights"""
        a_prev_bias = np.insert(np.mean(a_prev, axis=1), a_prev.shape[0], 1, axis = 0)
        return gradient.reshape(-1,1).dot(np.array([a_prev_bias]))
        
    def __cost_backward(self):
        """dC/da"""
        out = 2 * (self.__prediction - self.__y.to_numpy())
        gradient = np.mean(out, axis=0)
        return gradient
        
        
    def __activation_fun(self, x, deriv = False):
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
        elif self.__activation == 'relu':
            if deriv:
                #x[x<=0] = 0
                #x[x>0] = x -->???
                return None
            else:
                return np.max(0, x)
        else:
            raise Exception('no valid activation function, try: "tanh", "logistic" or "idendity"')
        
    def __sum_squared_errors(self):
        return np.sum((self.__prediction - self.__y.to_numpy())**2)
    
    def __J(self, thetas):
        #__forward with thetas as input??????
        return None
    
    def get_weights(self):
        return self.__weights
    
    def __euq_dist(self, x1, x2):
        x = x1 - x2
        return np.linalg.norm(x)
    
    def test_case_act_fun(self, X):
        matrix = X.to_numpy()
        res1 = self.__activation_fun(matrix, deriv = True)
        res2 = self.__deriv_approx(self.__activation_fun, matrix)
        res3 = self.__euq_dist(res1, res2)
        print('error act_fun:\n', res3)
    
    def __deriv_approx(self, f, x):
        h = 0.00001
        return (f(x+h) - f(x-h)) / (2*h)
    
    def eval_grad(f, x, verbose = False):
        """
        - f is a function
        - x is the input of the function f (numpy array)
        """
        h = 0.00001
        grad = np.zeros_like(x)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            orig_val = x[ix]
            x[ix] = orig_val + h
            f_p = f(x) # f(x + h)
            x[ix] = orig_val - h
            f_m = f(x) # f(x - h)
            x[ix] = orig_val #restore original value
            grad[ix] = (f_p - f_m) / (2*h)
            if verbose:
                print(ix, grad[ix])
            it.iternext()
        
        return grad
            
            
    
if __name__ == '__main__':
    nn = NeuralNetwork(hidden_layer_sizes = (5, 10,), random_state = 123, activation = 'logistic', batch_size = 4)
    d = {'col1': [1, 2, 3, 2], 'col2': [4, 5, 6, 1]}
    X = pd.DataFrame(data=d)
    d_y = {'col_y': [9, 10, 11, 4]}
    y = pd.DataFrame(data=d_y)
    nn.fit(X, y)
    #nn.test_case_act_fun(X)
    #predict_obj = {'col1': [1,6], 'col2': [2,5]}
    #print(nn.predict(pd.DataFrame(data = predict_obj)))
    
    
    
    
    
    
    
    
    
    
    
    
    