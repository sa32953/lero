##########################################
##       Created by: SAHIL ARORA        ## 
##  HOCHSCHULE- RAVENSBURG WEINGARTEN   ##
##          Date: 05.09.2020            ##
##########################################

# Dependencies imported

# For vector computations and notations
import numpy as np 

# For Plotting
import matplotlib.pyplot as plt

# Defining Solving Parameters

alpha = 0.01
error = 10 ** -4


############################
##       Read Data        ##
############################

# file in the same directory
data = np.loadtxt('data.txt', delimiter=',' )

# Initiate a Empty Input feature
x = np.array([])

# Fill values from read-data
for each in data:
    x = np.append(x , [each[0]])

# Initiate a Empty Output feature
y = np.array([])

for each in data:
    y = np.append(y , [each[1]])

############################
##    Visualize Data      ##
############################

plt.figure()
plt.plot(x,y,'ro', ms=10, mec='k')
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
#plt.show()

# Size of data
m = y.size

############################
##      Arrange Data      ##
############################

# Make a linear feature vector
x = np.stack([x, np.ones(m)], axis=1)

# Initiate weight vector
w = np.array([0,0])


############################
##   Gradient Function    ##
##  (Sqaured Loss Func)   ##
############################

def sq_gradient(x,w,y):
    
    # Valid for sqaure loss only
    # See analytical solution
    sq_grad = x*(np.dot(x,w) - y) 
    
    return sq_grad


############################
##    Gradient Descent    ##
############################

def gradient_decent():
    global itr, w, m, error

    delta_w = np.array([1,1]) # initialized randomly

    itr = 0
    while all(error < abs(a) for a in delta_w):
        
        sq_g = 0

        # Compute Cumulative gradient
        for a in range(len(y)):

            # Gradient computation is Normalized by number of data points available
            sq_g = sq_g + sq_gradient(x[a],w,y[a])/m
            a = a+1
        
        delta_w = alpha * sq_g
        # alpha is learning rate
        w = w - (delta_w)
        
        itr = itr+1
    
    return w, itr


sq_weight, sq_itr = gradient_decent()
print('The optimized weight vector is {}.'.format(sq_weight))
print('Solving criteria with Sq Loss Func: Convergency = {} and Learining Rate = {}'.format(error,alpha))
print('Total iterations done = {}'.format(sq_itr))


############################
##  Plot Regression Line  ##
############################

plt.plot(x[:, 0], np.dot(x, sq_weight), '-')
plt.legend(['Training data', 'Linear regression'])
plt.show()

