# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 20:28:13 2020

@author: Lukas
"""
import numpy as np

grad = np.array([2,3,6])

def f(x):
    return 2*x[0] + 3*x[1] + 6*x[2] + 7

def test_f(sample_size = 300):
    X = np.random.normal(5, 2, sample_size*3).reshape(sample_size, 3)
    y = [f(x) for x in X]
    grad = check_grad(f, np.array([2, 3, 4]), True)


def check_grad(f, x, verbose = False):
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
    test_f(10)
    l = [2, 3, 4, 5]
    print(list(reversed(l)))