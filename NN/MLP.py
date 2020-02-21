import numpy as np
import sys
import pickle
import random
import plotly.graph_objs as go
from IPython.display import display
from scipy.special import expit
from sklearn.preprocessing import LabelBinarizer

class MLP(object):
    ''' A Multi-Layer Perceptron.
    '''

    def __init__(self, layers, weight_init_int=(-.7, .7), max_iter=1000,
            learning_rate=0.01, epsilon=0.01, alpha=0., batch_size = 'auto', plot_error = False):
        '''
        layers: tuple
            The elements of the tuple define the number of units in all hidden
            layers. Bias units not included.

        weight_init_int: tuple =(-.7, .7)
            The interval on which the weights/thetas will be randomly initialized.

        epsilon: float
            The threshold value for the length of the gradient for stopping gradient
            descent iterations.
            
        learning_rate: float
            The (initial) step size.
            
        max_iter: int
            The maximal number of gradient descent iterations.

        alpha: float
            The l2 regularization strength. (ridge regression)
        '''
        # the model
        self.__layers = layers
        self.__weight_init_int = weight_init_int
        self.__alpha = alpha
        # gradient decscent params
        self.__epsilon = epsilon
        self.__max_iter = max_iter
        self.__learning_rate = learning_rate
        self.__batch_size = batch_size
        self.__plot_error = plot_error

        print('MLP(layers={}, alpha={}, learning_rate={})'.format(
            self.__layers, self.__alpha, self.__learning_rate,))
    
    def fit(self, X, y):
        '''
        Configures input and output layer, initializes weights and fits the
        model coefficients.
        '''
        if self.__plot_error:
            f = go.FigureWidget()
            f.add_scatter(y = [])
            f.update_layout(title = 'Network Error', xaxis_title = 'epochs', yaxis_title = 'SSE')
            display(f)
        # initialize the entire network, including input and ouput layer
        y_ = self._init_network(X, y)

        costs_ = [sys.float_info.max]
        normgrad = None
        
        # initialize batch_size if not set in __init__
        if self.__batch_size == 'auto':
            self.__batch_size = min(200, X.shape[0])
        elif self.__batch_size == 'none': #no mini-batch gradient descent
            X_batch, y_batch = X, y_

        # do gradient descent iterations
        numit = 0
        for numit in range(self.__max_iter):
            #create mini_batch
            if self.__batch_size != 'none':
                idx = random.sample(range(X.shape[0]), self.__batch_size)
                X_batch, y_batch = X[idx], y_[idx]
            if numit % 10 == 0:
                sys.stdout.flush()
                sys.stdout.write("\r" + 'num its = {} \t\t normgrad = {} \t\t cost = {}'.format(numit, normgrad, costs_[-1]))
                if self.__plot_error:
                    scatter = f.data[0]
                    if numit <= 20:
                        scatter.y = costs_[1:]
                    else:
                        scatter.y = costs_[21:]
            # get batch gradient
            grad = MLP.gradient_cost_function(self.__theta, self.__alpha, X_batch, y_batch)
            # update weights / thetas
            for tidx, theta in enumerate(self.__theta):
                theta -= self.__learning_rate*grad[tidx]
            costs_.append(MLP.cost_function(self.__theta, self.__alpha, X, y_))
            normgrad = np.linalg.norm(MLP.unroll(grad))
            if normgrad < self.__epsilon and costs_[-1] > costs_[-2]: #stops optimize early if gradient is short and the cost increased
                break

        self.__numits_ = numit
        self.__costs_ = np.array(costs_)

        print('\nfit completed!')
        print('number of iterations: ', self.__numits_)
        print('max iterations: ', self.__max_iter)
        print('norm(grad): ', normgrad)
        print('cost: ', costs_[-1])

    @staticmethod
    def cost_function(theta_, alpha, X, y, theta_shapes=None):
        '''Computes the cost function value.

        Uses MLP.forward_propagation

        Arguments
        ---------
        theta_ : the weights of the neural network
        alpha : the regularization strength
        X, y : the data
        theta_shapes : a list of tuples defining the shapes of theta

        Returns
        -------
        J : cost function value given thetas
        '''
        ### FILL IN CODE HERE 
        theta_ = MLP.rollup_if(theta_, theta_shapes)
        a = np.asarray([MLP.forward_propagation(theta_, x) for x in X])
        y_hat = a[:,-1]
        cost = np.sum((y_hat - y.T)**2) + alpha * np.sum(MLP.unroll(theta_)**2)
        return cost

    @staticmethod
    def gradient_cost_function(theta_, alpha, X, y, theta_shapes=None):
        '''Computes the gradient of the cost function.
        
        Uses MLP.forward_propagation and MLP.back_propagation.

        Arguments
        ---------
        theta_ : the weights of the neural network
        theta_shapes : a list of tuples defining the shapes of theta
        alpha : the regularization strength
        X, y : the data

        Returns
        -------
        grad : the gradient, unrolled
        '''
        ### FILL IN CODE HERE
        theta_ = MLP.rollup_if(theta_, theta_shapes) 
        a = np.asarray([MLP.forward_propagation(theta_, x) for x in X]) #elementwise forwardprop
        DELTA = MLP.back_propagation(theta_, a, y)
        DELTA_unrolled = MLP.unroll(DELTA) + alpha * 2 * MLP.unroll(theta_) #ridge regression (alpha * 2*theta_i)
        DELTA = MLP.rollup(DELTA_unrolled, MLP._get_theta_shapes(theta_))
        return DELTA 

    @staticmethod
    def forward_propagation(theta, x):
        '''Computes the activations for all units in an MLP given by theta for
        a single data point x.
        '''
        ### FILL IN CODE HERE 
        a, i_out, classify = x, len(theta)-1, False
        if len(theta[i_out]) > 1:
            classify = True
            
        for i in range(len(theta)):
            x = np.insert(x, x.shape[0], 1)
            x = theta[i].dot(x)
            a = np.concatenate((a, x), axis = None)
            if (i == i_out) and not classify:
                continue
            else:
                x = MLP.phi(x)
                a = np.concatenate((a, x), axis = None)
        return a

    @staticmethod
    def back_propagation(theta, a, y):
        '''Computes the error d for all units. '''
        ### FILL IN CODE HERE 
        a = (a.T).copy() #transpond activations for access simplicity
        d, n_train, n_out, n_second_last = [], a.shape[1], theta[-1].shape[0], theta[-2].shape[0]
        tmp = 2*(a[-n_out:].T - y) #len(theta[-1]) provides the numbers of output neurons
        a = a[:-n_out] #update unused activations
        if n_out > 1: #in case of classification
            tmp *= MLP.phi_der(a[-n_out:].T)
            a = a[:-n_out]
        tmp = tmp.T
        a_L = a[-n_second_last:]
        a = a[:-n_second_last]
        a_L = np.insert(a_L, a_L.shape[0], 1, axis = 0) #add bias activations
        grad = np.dot(tmp, a_L.T) * (1/n_train) #dz/dw
        d.append(grad)
        
        for i in range(1, len(theta)):
            n = theta[-i].shape[1] - 1 #number of neurons in L (-1 is to not consider bias vector)
            tmp = np.dot((theta[-i][:,:-1]).T, tmp) #dz/da^(L-1), [:,:-1] to not consider biases
            tmp *= MLP.phi_der(a[-n:]) #da/dz
            a = a[:-n]
            n_prev = theta[-(i+1)].shape[1] - 1
            a_L = a[-n_prev:]
            a = a[:-n_prev]
            a_L = np.insert(a_L, a_L.shape[0], 1, axis = 0)
            grad = np.dot(tmp, a_L.T) * (1/n_train) #dz/dw
            d.append(grad)
            
        return d[::-1] # reverse order if required, ie [::-1]

    def grad_check(self, X, y, epsilon=0.0001, decimal=3, verbose=False, alpha = 0.):
        '''Compare the gradient with finite differences around current point
        in parameter space.
        '''
        theta_ur = MLP.unroll(self.__theta)
        y = MLP._trans_y(y)
        # approximate the gradient by finite differences
        approxgrad = []
        for idx in range(len(theta_ur)):
            # modify theta[idx] +/- epsilon
            tplus = theta_ur.copy()
            tplus[idx] = tplus[idx]+epsilon
            pluseps = MLP.cost_function(tplus, alpha, X, y, self.__theta_shapes)
            tminus = theta_ur.copy()
            tminus[idx] = tminus[idx]-epsilon
            # calculate the costfunctions
            minuseps = MLP.cost_function(tminus, alpha, X, y, self.__theta_shapes)
            pluseps = MLP.cost_function(tplus, alpha, X, y, self.__theta_shapes)
            # finite diffs
            approxgrad.append((pluseps - minuseps)/(2*epsilon))

        approxgrad = np.array(approxgrad)
        approxgrad /= np.linalg.norm(approxgrad)
        calcgrad = MLP.gradient_cost_function(theta_ur, alpha, X, y, self.__theta_shapes)
        calcgrad = MLP.unroll(calcgrad)
        calcgrad /= np.linalg.norm(calcgrad)

        if verbose:
            print('\napprox :\n', approxgrad)
            print('\ncalc :\n', calcgrad)

        np.testing.assert_array_almost_equal(approxgrad, calcgrad, decimal=decimal)

    def predict(self, X):
        '''Predicts the output for all data points in X.

        Makes use of MLP.forward_propagation
        '''
        ### FILL IN CODE HERE 
        a = np.asarray([MLP.forward_propagation(self.__theta, x) for x in X])
        y_hat = a[-1]
        return y_hat

    @staticmethod
    def rollup_if(x_, shapes):
        '''Conditional uprolling if shapes is not None.
        '''
        ### FILL IN CODE HERE 
        # conditional returns
        if shapes != None:
            return MLP.rollup(x_, shapes)
        return x_

    @staticmethod
    def unroll(xlist):
        '''Unrolling theta in a 1d array (that can be passed into minimize).
        '''
        ### FILL IN CODE HERE 
        x_unrolled = np.concatenate([x.ravel() for x in xlist], axis = None)
        return x_unrolled

    @staticmethod
    def rollup(x_unrolled, shapes):
        '''
        Rolling up theta into a list of 2d matrices.
        - shapes: (m1, n1, m2, n2, ...)
        '''
        ### FILL IN CODE HERE
        xlist = []
        for i in range(int(len(shapes)/2)):
            m, n = shapes[2*i], shapes[2*i+1]
            sub_theta = x_unrolled[:(m*n)]
            x_unrolled = x_unrolled[(m*n):]
            xlist.append(sub_theta.reshape(m, n))
        return xlist

    @staticmethod
    def phi(t):
        '''Logistic / sigmoid function.'''
        return expit(t)
        #return 1. / (1 + np.exp(-t)) #causes overflow problems for big negative numbers
    
    @staticmethod
    def phi_der(t):
        '''Logistic / sigmoid function derivative'''
        return MLP.phi(t) * (1 - MLP.phi(t))

    def _init_network(self, X, y):
        '''
        - transforms y if required (classification), and returns transformed y_
        - completes self.__layers
        - initializes thetas, using MLP.init_theta, as list of 2-d matrices
        - sets self.__theta_shapes (needed for unrolling and uprolling)
        '''
        ### FILL IN CODE HERE
        y_ = MLP._trans_y(y)
        self.__layers = (X.shape[1],) + self.__layers + (y_.shape[1],)
        self.__theta = MLP.init_theta(self.__layers, self.__weight_init_int)
        self.__theta_shapes = MLP._get_theta_shapes(self.__theta)
        return y_
    
    @staticmethod
    def _get_theta_shapes(theta):
        return tuple(np.asarray(list(t.shape for t in theta)).ravel())
    
    @staticmethod
    def _trans_y(y):
        '''Transforms y
        '''
        if y.dtype.type == np.str_:
            lb = LabelBinarizer()
            lb.fit(y)
            y_ = lb.transform(y)
        else:
            y_ = y.reshape(-1,1)
        return y_

    @staticmethod
    def init_theta(layers, weight_init_int):
        '''Initializes the thetas and returns them as a list of 2-d matrices.
        '''
        ### FILL IN CODE HERE
        w_fac, w_sum = abs(weight_init_int[0]) + abs(weight_init_int[1]), weight_init_int[0]
        theta = [np.random.rand(y, x+1)*w_fac+w_sum for x, y in zip(layers[:-1], layers[1:])] #x+1 is for bias purpose
        return theta
    
    def store(self, path):
        '''Stores the MLP as a pickle-object
        '''
        if len(path) < 4:
            path = path + '.pkl'
        elif path[-4:] != '.pkl':
            path = path + '.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod
    def load(path):
        '''Loads a stored pickle-MLP
        '''
        if len(path) < 4:
            path = path + '.pkl'
        elif path[-4:] != '.pkl':
            path = path + '.pkl'
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    
if __name__ == '__main__':
    '''
    - X: [[object_1], [object_2], ...]
    - y: [y1, y2, y3, ...]
    '''
    nn = MLP((10,), learning_rate = 0.1, batch_size = 'none', max_iter = 10000)
    #REGRESSION
    #X = np.array([[2.,3.], [3.,4.]])
    #y = np.array([9.,5.])
    
    X = np.random.rand(1000, 20)
    y = np.random.rand(1, 1000) * 50
    
    #MULTI CLASSIFICATION
    #X = np.array([[2.,3.], [3.,4.], [1.,2.], [4.,2.]])
    #y = np.array(['9.','7.','2.','9.'])
    
    #BINARY CLASSIFICATION
    #X = np.array([[2.,3.], [3.,4.], [1.,2.], [4.,2.]])
    #y = np.array(['9.','2.','2.','9.'])
    
    #nn.fit(X, y)
    #nn.grad_check(X, y, verbose = True)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
