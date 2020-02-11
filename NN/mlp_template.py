import numpy as np
import sys
import pickle

from sklearn.preprocessing import LabelBinarizer

class MLP(object):
    ''' A Multi-Layer Perceptron.
    '''

    def __init__(self, layers, weight_init_int=(-.7, .7), max_iter=1000,
            learning_rate=0.3, epsilon=0.01, alpha=0.):
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
            The l2 regularization strength.
        '''
        # the model
        self.__layers = layers
        self.__weight_init_int = weight_init_int
        self.__alpha = alpha
        # gradient decscent params
        self.__epsilon = epsilon
        self.__max_iter = max_iter
        self.__learning_rate = learning_rate

        print('MLP(layers={}, alpha={}, learning_rate={})'.format(
            self.__layers, self.__alpha, self.__learning_rate,))
    
    def fit(self, X, y):
        '''
        Configures input and output layer, initializes weights and fits the
        model coefficients.
        '''
        # initialize the entire network, including input and ouput layer
        y_ = self._init_network(X, y)

        costs_ = []
        normgrad = None

        # do gradient descent iterations
        for numit in range(self.__max_iter):
            if numit % 10 == 0:
                sys.stdout.flush()
                sys.stdout.write("\r" + 'num its = {} \t\t normgrad = {}'.format(numit, normgrad))
            # get gradient
            grad = MLP.gradient_cost_function(self.__theta, self.__alpha, X, y_)
            # update weights / thetas
            for tidx, theta in enumerate(self.__theta):
                theta -= self.__learning_rate*grad[tidx]
            costs_.append(MLP.cost_function(self.__theta, self.__alpha, X, y_))
            normgrad = np.linalg.norm(MLP.unroll(grad))
            if normgrad < self.__epsilon:
                break

        self.__numits_ = numit
        self.__costs_ = np.array(costs_)

        print('\nfit completed!')
        print('number of iterations: ', self.__numits_)
        print('max iterations: ', self.__max_iter)
        print('norm(grad): ', normgrad)

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
        a = np.asarray([MLP.forward_propagation(theta_, x) for x in X]) #elementwise forwardprop
        DELTA = MLP.back_propagation(theta_, a, y)
        return DELTA 

    @staticmethod
    def forward_propagation(theta, x):
        '''Computes the activations for all units in an MLP given by theta for
        a single data point x.
        '''
        ### FILL IN CODE HERE 
        a, i_out, classify = [], len(theta)-1, False
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
        d, n_out, n_second_last = [], theta[-1].shape[0], theta[-2].shape[0]
        print('\ntheta:\n', theta)
        tmp = 2*(a[-n_out:].T - y) #len(theta[-1]) provides the numbers of output neurons
        a = a[:-n_out] #update unused activations
        if n_out > 1: #in case of classification
            tmp *= MLP.phi_der(a[-n_out:].T)
            a = a[:-n_out]
        print('\ntmp:\n', tmp)
        print('\nactiv:\n', a[-n_second_last:])
        grad = np.dot(a[-n_second_last:], tmp) #dz/dw
        print('\ngrad:\n', grad)
        a = a[:-n_second_last]
        d.append(grad)
        print('\nactivations:\n', a)
        for i in range(2, len(theta)+1):
            print('\ncurrent_theta:\n', (theta[-i][:,:-1]))
            print('\ntmp:\n', tmp)
            n = theta[-i].shape[0] #number of neurons in current layer
            tmp = np.dot(theta[-i][:,:-1], tmp) #[:,:-1] to not consider biases
            tmp *= MLP.phi_der()
        return d[::-1] # reverse order if required, ie [::-1]

    def grad_check(self, X, y, epsilon=0.0001, decimal=3, verbose=False):
        '''Compare the gradient with finite differences around current point
        in parameter space.
        '''
        theta_ur = MLP.unroll(self.__theta)
        # approximate the gradient by finite differences
        approxgrad = []
        for idx in range(len(theta_ur)):
            # modify theta[idx] +/- epsilon
            tplus = theta_ur.copy()
            tplus[idx] = tplus[idx]+epsilon
            pluseps = MLP.cost_function(tplus, 0., X, y, self.__theta_shapes)
            tminus = theta_ur.copy()
            tminus[idx] = tminus[idx]-epsilon
            # calculate the costfunctions
            minuseps = MLP.cost_function(tminus, 0., X, y, self.__theta_shapes)
            pluseps = MLP.cost_function(tplus, 0., X, y, self.__theta_shapes)
            # finite diffs
            approxgrad.append((pluseps - minuseps)/(2*epsilon))

        approxgrad = np.array(approxgrad)
        approxgrad /= np.linalg.norm(approxgrad)
        calcgrad = MLP.gradient_cost_function(theta_ur, 0., X, y, self.__theta_shapes)
        calcgrad /= np.linalg.norm(calcgrad)

        if verbose:
            print('approx : ', approxgrad)
            print('calc : ', calcgrad)

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
        return 1. / (1 + np.exp(-t))
    
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
        y_shape = 1 #y_shape of 1 assumes regression
        
        #transforms y if classification
        if y.dtype.type == np.str_:
            lb = LabelBinarizer()
            lb.fit(y)
            y_ = lb.transform(y)
            y_shape = len(lb.classes_)
        else:
            y_ = y.reshape(-1,1)
        
        self.__layers = (X.shape[1],) + self.__layers + (y_shape,)
        self.__theta = MLP.init_theta(self.__layers, self.__weight_init_int)
        self.__theta_shapes = [t.shape for t in self.__theta]
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
    nn = MLP((3,))
    #REGRESSION
    X = np.array([[2.,3.], [3.,4.], [1.,5.]])
    y = np.array([9.,7.,8.])
    
    #MULTI CLASSIFICATION
    #X = np.array([[2.,3.], [3.,4.], [1.,2.], [4.,2.]])
    #y = np.array(['9.','7.','2.','9.'])
    
    #BINARY CLASSIFICATION
    #X = np.array([[2.,3.], [3.,4.], [1.,2.], [4.,2.]])
    #y = np.array(['9.','2.','2.','9.'])
    
    theta = nn.fit(X, y)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
