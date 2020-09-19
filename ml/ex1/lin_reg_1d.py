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

def gradient_decent(x,y,w,acc,alpha):
    global itr, m, error

    delta_w = np.array([1,1]) # initialized randomly

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

print('Solving...........')
alphas = [0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175, 0.02]

sq_weight = [0] * len(alphas)
sq_itr = [0] * len(alphas)

for b in range(len(alphas)):
    sq_weight[b], sq_itr[b] = gradient_decent(x,y,w,acc,alphas[b])


############################
##  Plot Iteration Var.   ##
############################

print('Solved. See plot for variation.')
plt.plot(alphas, sq_itr, '-')
plt.ylabel('Number of Iterations for convergance')
plt.xlabel('Learning Rate')
plt.grid(b=None, which='major', axis='both')
plt.legend(['Convergence Iterations Vs Learning Rate'])
plt.xticks(np.linspace(0.005, 0.02, 7))
plt.show()

