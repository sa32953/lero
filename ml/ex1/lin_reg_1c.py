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

alpha = 0.015
acc = 10 ** -4
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
##  (Absolute Loss Func)  ##
############################

def ab_gradient(x,w,y):
    
    # Valid for Absolute loss only
    # See analytical solution
    ab_grad = x*np.sign((np.dot(x,w) - y))
    
    return ab_grad

############################
##    Gradient Descent    ##
############################

def gradient_decent(x,y,w,acc):
    global itr, m, error

    delta_w = np.array([1,1]) # initialized randomly

    itr = 0
    while all(acc < abs(a) for a in delta_w):
        
        ab_g = 0

        # Compute Cumulative gradient
        for a in range(len(y)):

            # Gradient computation is Normalized by number of data points available
            ab_g = ab_g + ab_gradient(x[a],w,y[a])/m
            a = a+1
        
        delta_w = alpha * ab_g
        # alpha is learning rate
        w = w - (delta_w)
        
        itr = itr+1
    
    return w, itr


ab_weight, ab_itr = gradient_decent(x,y,w,acc)
print('The optimized weight vector is {}.'.format(ab_weight))
print('Solving criteria with Abs Loss Func: Convergency = {} and Learining Rate = {}'.format(acc,alpha))
print('Total iterations done = {}'.format(ab_itr))


#In earlier part
sq_weight = np.array([1.13774908,-3.34547133])

############################
##  Plot Regression Line  ##
############################

plt.plot(x[:, 0], np.dot(x, ab_weight), '-')
plt.plot(x[:, 0], np.dot(x, sq_weight), '.')
plt.legend(['Training data', 'Ab_Linear regression', 'Sq_Linear regression'])
plt.show()

