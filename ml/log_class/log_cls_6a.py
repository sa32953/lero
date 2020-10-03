##########################################
##  HOCHSCHULE- RAVENSBURG WEINGARTEN   ##
##          Date: 25.09.2020            ##
##########################################

# Dependencies imported

# For vector computations and notations
import numpy as np 
# For Plotting
import matplotlib.pyplot as plt
import pandas as pd

# Defining Solving Parameters
alpha = 0.8
acc = 10 ** -2

############################
##       Read Data        ##
############################

# file in the same directory
data = pd.read_csv('/home/sa0102/robot_learning/excercises/ml/log_class/log_class_data.txt', names=['Exam 1', 'Exam 2', 'Admitted'])

# export pandas dataframe to numpy array
X = data[['Exam 1','Exam 2']].values 
y = data['Admitted'].values


# Regressions with logistics loss work with Classification Outputs as {-1,1}. Change the output..
#.. label of y = 0 to y = -1

for b in range(len(y)):
    if y[b]==0: # for y = 0 labels only
        y[b]=-1 # change to y = -1


############################
##    Visualize Data      ##
############################

# Find Indices of Positive and Negative Examples, to visualize separately
pos = y == 1
neg = y == -1

# Plot Examples
plt.plot(X[pos, 0], X[pos, 1], 'k*', mfc='r',ms=10)
plt.plot(X[neg, 0], X[neg, 1], 'ko', mfc='g', ms=8)
plt.grid(b=None, which='major', axis='both')
plt.legend(['Admitted', 'Rejected'])
plt.title('University Database')
plt.xlabel('Exam 1')
plt.ylabel('Exam 2')
#plt.show()

############################
##      Arrange Data      ##
############################

# add bias
X = np.hstack((X,np.ones((X.shape[0],1))))

#Initiate randomized weight verctor
w = np.array([0,0,0])

# Size of data
m = y.size


############################
##    Hypothesis Func     ##
############################

# We work with linear hypothesis in this program and later will see also Sigmoid hypothesis.

def h(w,x): 
    
    return np.dot(x,w)

#return sigmoid( np.dot(x,w.T) )


############################
##   Gradient Function    ##
##    (Log Loss Func)     ##
############################

def log_gradient(x,w,y):

    # Valid for Logarithmic Loss func only
    # See analytical solution in slide

    log_grad = -y*x/(1+ np.exp(y*h(w,x)))
    
    return log_grad

# In the next part we will work with the Sigmoid Loss function
#log_grad =  - np.dot( y-h(w,x), x) / m

############################
##    Gradient Descent    ##
############################

def gradient_decent(x,y,w,acc):
    global m, itr

    delta_w = np.array([1,1,1]) # initialized randomly

    itr = 0
    while all(acc < abs(a) for a in delta_w):
        
        log_g = 0

        # Compute Cumulative gradient
        for a in range(m):

            # Gradient computation is Normalized by number of data points available
            log_g = log_g + log_gradient(X[a],w,y[a])/m
            a = a+1
        
        # Modify weight
        delta_w = alpha * log_g
        # alpha is learning rate
        w = w - (delta_w)
        
        itr = itr+1
        
    return w, itr

# Display the obtained results

log_weight, log_itr = gradient_decent(X,y,w,acc)

print('The optimized weight vector is {}.'.format(log_weight))
print('Solving criteria with Sq Loss Func: Convergency = {} and Learining Rate = {}'.format(acc,alpha))
print('Total iterations done = {}'.format(log_itr))

################################
##  Plot Classification Line  ##
################################

# It is enough to compute for 2 points in order to plot the line. 
# We can do it by max and min points

# Find min and max points
x_min, x_max = X[:, 0].min(), X[:, 0].max()
plot_x1 = np.array([x_min, x_max])

# Compute the y axis point from the learned weights
plot_x2 = (-1 / log_weight[1]) * (log_weight[0] * plot_x1 + log_weight[2])

plt.plot(plot_x1, plot_x2)
plt.show()