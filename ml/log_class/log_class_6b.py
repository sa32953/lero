##########################################
##  HOCHSCHULE- RAVENSBURG WEINGARTEN   ##
##          Date: 25.09.2020            ##
##########################################

# Dependencies imported

# For vector computations and notations
import numpy as np 
import pandas as pd

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

############################
##      Unknown Data      ##
############################

# Suppose the learned weight is calculated from part A of this problem
log_weight = np.array([22.07838229,    13.6649195,  -2197.9842622])

# We store the new unknown data as feature vector. 
x_new = np.array([70,47,1])

# To find the class of new point, we need to compute the hypothesis value.
# Class is dependent on the sign of the obtained value.
cat = np.sign(np.dot(log_weight, x_new))

# Relate the sign of the value to a category
# Class = 1 = Admitted, Class = -1 = Rejected 

if cat == 1:
    print('Student Admitted; Class 1')

if cat == -1:
    print('Student Rejected; Class -1')
