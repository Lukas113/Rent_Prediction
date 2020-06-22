# ImmoRent: Rental price estimate for real estate in Switzerland

Studierender: Lukas Gehrig  
Fachcoach: Michael Graber  

# Neural Network

The class NeuralNetwork provides a feedforward Neural Network for regression in scikit format which is easy memory and loadable.

- example
layers, learning_rate, alpha, batch_size, max_iter, plot_error = (80,), 0.01, 0.001, 'none', 1000, True
mlp = MLP(layers, learning_rate = learning_rate, alpha = alpha, batch_size = batch_size, max_iter = max_iter, plot_error = plot_error)
mlp.fit(X_train, y_train)
predicted = mlp.predict(X_test)
    
m = mape(y_test, predicted)
residuals = y_test - predicted

- store
mlp.store('path.pkl')

