##########################################
##       Created by: SAHIL ARORA        ## 
##  HOCHSCHULE- RAVENSBURG WEINGARTEN   ##
##          Date: 15.09.2020            ##
##########################################

# Dependencies imported

# For vector computations and notations
import numpy as np 

# For Plotting
import matplotlib.pyplot as plt

# Defining Solving Parameters

# Range 0.0001 to 0.00001
alpha = 1 * 10 ** -4
acc = 10 ** -5

degree = 2


############################
##       Read Data        ##
############################

# file in the same directory
data = np.loadtxt('data.txt', delimiter=',' )

# Initiate a Empty Input feature
x_data = np.array([])

# Fill values from read-data
for each in data:
    x_data = np.append(x_data , [each[0]])

# Initiate a Empty Output feature
y_data = np.array([])

for each in data:
    y_data = np.append(y_data , [each[1]])

############################
##    Visualize Data      ##
############################

plt.figure()
plt.plot(x_data , y_data ,'ro', ms=10, mec='k')
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
#plt.show()

# Size of data
m = y_data.size


############################
##    Feature Function    ##
############################

def feature(x):
    global degree
    # produces a feature vector with given inputs 

    phi_temp = np.array([])
    
    # depending on degree, make a feauture vector with powers of input x
    for p in range(degree + 1):
        phi_temp = np.append(phi_temp, [x ** p]) # storing in an array

    return phi_temp


############################
##    Hypothesis Func     ##
############################

def hypo(ix,w):
    # function for dot product
    h = np.dot(ix,w)

    return h

############################
##   Gradient Function    ##
##  (Sqaured Loss Func)   ##
############################

def sq_gradient(x,w,y):
    
    # Valid for sqaure loss only
    # See analytical solution
    
    sq_grad = feature(x)*(hypo(feature(x), w) - y) 
    
    return sq_grad


############################
##    Gradient Descent    ##
############################

def gradient_decent(x,y,w,acc):
    global itr, m

    delta_w = np.ones(degree + 1) # initialized randomly

    itr = 0
    while all(acc < abs(a) for a in delta_w):
        
        sq_g = 0

        # Compute Cumulative gradient
        for a in range(m):

            # Gradient computation is Normalized by number of data points available
            sq_g = sq_g + sq_gradient(x[a],w,y[a])/m
            a = a+1
       
        delta_w = alpha * sq_g 
        
        # alpha is learning rate
        w = w - (delta_w)
        
        itr = itr+1
    
    return w, itr


weight = np.zeros(degree+1)

sq_weight, sq_itr = gradient_decent(x_data, y_data, weight, acc)
print('The optimized weight vector is {}.'.format(sq_weight))
print('Solving criteria with Sq Loss Func: Convergency = {} and Learining Rate = {}'.format(acc,alpha))
print('Total iterations done = {}'.format(sq_itr))


############################
##  Plot Regression Line  ##
############################

# sorint to get a continuous polynomial output
x_data.sort()

new_y = np.array([])
for every in x_data:
    new_y = np.append(new_y, [hypo(feature(every), sq_weight)]) 

plt.plot(x_data, new_y, '-')
plt.legend(['Training data', 'Non - Linear regression'])
plt.show()

