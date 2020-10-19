########################################## 
##  HOCHSCHULE- RAVENSBURG WEINGARTEN   ##
##          Date: 15.10.2020            ##
##########################################

# Used for manipulating directory paths
import os

# Scientific and vector computation for python
import numpy as np

# Plotting library
from matplotlib import pyplot

# Optimization module in scipy
from scipy import optimize

# will be used to load MATLAB mat datafile format
from scipy.io import loadmat

from PIL import Image


#  training data stored in arrays X, y
data = loadmat('ex3data1.mat')

X, y = data['X'], data['y'].ravel()
m = y.size

# Setting the zero digit to 0.
# This is an artifact due to the fact that this dataset was used in...
# ...MATLAB where there is no index 0
y[y == 10] = 0

# Choose a random index
rand_indices = np.random.choice(m, 1)


def show_img(a):

    # Select the row from X matrix with the specified argument
    selc = X[a, :]

    # Unflatten the row to original 20x20 size
    selc = np.reshape(selc, (20,20), order='F')
    
    # Multiply by 255 to get a grayscale range in 0 to 255
    im = Image.fromarray(selc*255)

    # Increase the size for better visualization
    size = (512,512)
    im = im.resize(size)
    im.show()


show_img(rand_indices)

# Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 0 to 9

# Load the .mat file, which returns a dictionary 
weights = loadmat('ex3weights.mat')

# get the model weights from the dictionary
# Theta1 has size 25 x 401
# Theta2 has size 10 x 26
Theta1, Theta2 = weights['Theta1'], weights['Theta2']

# swap first and last columns of Theta2, due to legacy from MATLAB indexing, 
# since the weight file ex3weights.mat was saved based on MATLAB indexing
Theta2 = np.roll(Theta2, 1, axis=0)


def sigmoid(z):
    """
    Computes the sigmoid of z.
    """
    return 1.0 / (1.0 + np.exp(-z))

def predict(t1,t2,X):
    # The prediction function computes the probability of every label as described in theory
    # Find the index of the max value to give its class.
    
    # f1 function
    z2 = np.dot(Theta1[:,1:],np.transpose(X).flatten())+ Theta1[:,0]
    a2 = sigmoid(z2)

    # f2 function
    z1 = np.dot(Theta2[:,1:],a2)+ Theta2[:,0]
    a1 = sigmoid(z1)
    
    return np.argmax(a1), a1[np.argmax(a1)]


# Predict a class of a random index
klass, probability = predict(Theta1, Theta2, X[rand_indices,:])
print('The predicted class of the given output is {} with probability = {}'.format(klass, probability))
print('The true class of the given output wrt dataset is {}.'.format(y[rand_indices]))

err = 0 # Initiating variable

# Predict for every
for k in range(m):
    klass_pred, pred = predict(Theta1, Theta2, X[k,:])
    klass_act = y[k]
    
    # if incorrect prediction, then count as inaccuracy
    if klass_pred != klass_act:
        err = err + 1
        

print('Accuracy of trained model is {}%'.format((1-err/m)*100))