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
from sklearn import model_selection

# Defining Solving Parameters

alpha = 1 * 10 ** -1
acc = 10 ** -5


############################
##       Read Data        ##
############################

# file in the same directory
data = np.genfromtxt('nl_data.csv', delimiter=',' )

train_data , test_data = model_selection.train_test_split(data)

# Initiate a Empty Input feature
x_data = np.array([])

# Fill values from read-data
for each in train_data:
    x_data = np.append(x_data , [each[0]])

# Initiate a Empty Output feature
y_data = np.array([])

for each in train_data:
    y_data = np.append(y_data , [each[1]])

# Size of data
m = y_data.size


############################
##    Noramalize data     ##
############################

# Finding the max and min to normalize the data.
max_x = max(x_data)
min_x = min(x_data)
max_y = max(y_data)
min_y = min(y_data)


# Each data point in X and Y are normalized as follows
# New_x = (Old_x - Min_x) / (Max_x - Min_x) 

for f in range(m):
    x_data[f] = (x_data[f] - min_x) / (max_x - min_x)

for h in range(m):
    y_data[h] = (y_data[h] - min_y) / (max_y - min_y)



############################
##    Feature Function    ##
############################

def feature(x,degree):
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

def sq_gradient(x,w,y,degree):
    
    # Valid for sqaure loss only
    # See analytical solution
    
    sq_grad = feature(x,degree)*(hypo(feature(x,degree), w) - y) 
    
    return sq_grad


############################
##    Gradient Descent    ##
############################

def gradient_decent(x,y,w,acc,degree):
    global itr, m

    delta_w = np.ones(degree + 1) # initialized randomly

    itr = 0
    while all(acc < abs(a) for a in delta_w):
        
        sq_g = 0

        # Compute Cumulative gradient
        for a in range(m):

            # Gradient computation is Normalized by number of data points available
            sq_g = sq_g + sq_gradient(x[a], w, y[a], degree)/m
            a = a+1
       
        delta_w = alpha * sq_g 
        
        # alpha is learning rate
        w = w - (delta_w)
        
        itr = itr+1
    
    return w, itr


############################
##     Bunch Solving      ##
############################

degree = [ 2, 3, 4, 6, 8, 10, 12, 15, 18, 20, 22, 24, 26, 28, 30]

sq_weight = [0] * len(degree)
sq_itr = [0] * len(degree)

print('Solving.............')
for b in range(len(degree)):

    weight = np.zeros(degree[b] + 1)

    sq_weight[b], sq_itr[b] = gradient_decent(x_data, y_data, weight, acc, degree[b])

print('The weight vectors are optimized for each degree specification.')

# Uncomment the next line to print weight vector of a specific degree. Carefull with indexing.
#print('The optimized weight vector is {}.'.format(sq_weight[i]))

print('Solving criteria with Sq Loss Func: Convergency = {} and Learining Rate = {}'.format(acc,alpha))
print('Total iterations done = {}. (In same order as degree specification)'.format(sq_itr))


############################
##        Test data       ##
############################

# Initiate a Empty Input feature
tx_data = np.array([])

# Fill values from read-data
for each in test_data:
    tx_data = np.append(tx_data , [each[0]])

# Initiate a Empty Output feature
ty_data = np.array([])

for each in test_data:
    ty_data = np.append(ty_data , [each[1]])

# Normalize Test data with concept as applied in Training Data
max_tx = max(tx_data)
min_tx = min(tx_data)
max_ty = max(ty_data)
min_ty = min(ty_data)

for f in range(len(ty_data)):
    tx_data[f] = (tx_data[f] - min_tx) / (max_tx - min_tx)

for h in range(len(ty_data)):
    ty_data[h] = (ty_data[h] - min_ty) / (max_ty - min_ty)



############################
##      Error function    ##
############################

def errors(y,yp):
    # Computes the mean error between Actual Output Value vs the Predicted Value
    total_error = 0

    for r in range(len(y)):
        total_error = total_error + ( ( y[r] - yp[r] ) ** 2 )

    mean_err = total_error / (len(y))

    return mean_err


############################
##    Plot Error Curve    ##
############################

# Compute Output based on Learned Weight vector
new_y = [np.array([])]*len(degree)

for k in range(len(degree)):
    # for each degree
    for every in tx_data:
        new_y[k] = np.append(new_y[k], [hypo(feature(every,degree[k]), sq_weight[k])]) 

mean_err = [np.array([])]*len(degree)

for u in range(len(degree)):
    mean_err[u] = errors(ty_data, new_y[u])

plt.plot(degree,mean_err)

plt.xlabel('Degree of learned non-linear configuration')
plt.ylabel('Mean error on Unseen Data')
plt.grid(b=None, which='major', axis='both')
plt.legend(['Error Curve'])
plt.xticks(np.linspace(1,30,30))
plt.show()

