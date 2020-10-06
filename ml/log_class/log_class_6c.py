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


###################################
##  Model's Prediction Accuracy  ##
###################################

# Suppose the learned weight is calculated from part A of this problem
log_weight = np.array([22.07838229,    13.6649195,  -2197.9842622])

# Compute the prediction of each Input Point and store in a new array.
y_new = np.array([])

for each in X:
    y_temp = np.sign(np.dot(log_weight, each))
    y_new = np.append(y_new, [y_temp])

# Compare the predicted value with the Actual Output label class. If doesn't match, consider...
#... this as wrong prediction
wrong = 0

for p in range(m):
    if y[p] != y_new[p]:
        wrong = wrong + 1

# Accuracy = (Correct_predictions / Total Predictions) * 100 %
print('Accuracy of Learned model is {} %.'.format(100- (100*wrong/m)))