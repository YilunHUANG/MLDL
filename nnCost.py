import numpy as np

def nnCost(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,lambda)
    '''
    two layer neural network
    X: training example
    lambda: regularization parameter
    '''

    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].\
             reshape(hidden_layer_size,(input_layer_size+1))

    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):].\
             reshape(num_labels,(hidden_layer_size+1))

    m,n = X.shape
    cost = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
