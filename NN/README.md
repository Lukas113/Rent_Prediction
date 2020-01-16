# ImmoRent: Rental price estimate for real estate in Switzerland

Studierender: Lukas Gehrig  
Fachcoach: Michael Graber  

# Neural Network

The class NeuralNetwork provides a feedforward Neural Network for regression in scikit format which is easy memory and loadable.

- example
nn = NeuralNetwork(hidden_layer_sizes = (50, 20,), activation = 'logistic', batch_size = 100)
nn.fit(X, y)
predictions = nn.predict(X)

- store
nn.store(path)
nn = NeuralNetwork.load(path)

The NN does not work yet, because the calculated gradient of the NN and the approximated gradient do not match.

Testing can be done at the very bottom of NeuralNetwork.py in the Main function. This includes, for example, the gradient check.