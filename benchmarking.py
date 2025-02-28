# Benchmark training time for knn, svc and dtree models

import pandas as pd
import time

iterations = 10

def benchmark(model, X_train, X_test, y_train, y_test, wintitle='unknown'):
  print ('\n\n' + wintitle + ' Results')
  s = time.time()
  for i in range(iterations):
    #
    # train the classifier on the training data / labels:
    # 
    model.fit(X_train, y_train)
  print ("{0} Iterations Training Time: ".format(iterations), time.time() - s)


  s = time.time()
  for i in range(iterations):
    #
    # score the classifier on the testing data / labels:
    # 
    score = model.score(X_test, y_test)
  print ("{0} Iterations Scoring Time: ".format(iterations), time.time() - s)
  print (" Score: ", round((score*100), 3))

#import os
#os.chdir("Documents/DS/Dat210x/")
X = pd.read_csv("Datasets/wheat.data")

# We can check which rows have nans in them
#print X[pd.isnull(X).any(axis=1)]
X = X.dropna(axis=0)

# 
# Nan handling can also be done by setting the nan values to the
# mean value of that column (axis=1)

#
# Since we are predicting wheat_type, we'll copy the labels out of the dataset into 
# a variable 'y', then Remove the labels from X. We'll encode the labels, using the .map() 
# -- canadian:0, kama:1, and rosa:2
# 
yc = X['wheat_type'].copy()
X.drop(labels=['wheat_type', 'id'], inplace=True, axis=1)
y = yc.map({'canadian':0, 'kama':1, 'rosa':2})


# 
# Next we'll split the data into test / train sets
# test size can be 30% with random_state 7.
# We'll use variable names: X_train, X_test, y_train, y_test
# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)

#
# Let's create an SVC classifier named svc
# and use a linear kernel, and set the C value
#

from sklearn.svm import SVC
svc = SVC(kernel='linear', C=1)

svc.fit(X_train, y_train) 
score_svc = svc.score(X_test, y_test)
#print("svc_score is " + str(score_svc))
benchmark(svc, X_train, X_test, y_train, y_test, 'SVC')

#
# Let's create an KNeighbors classifier named knn
# and set the neighbor count to 5
#

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train) 
score_knn = knn.score(X_test, y_test)
#print("knn_score is " + str(score_knn))
benchmark(knn, X_train, X_test, y_train, y_test, 'KNeighbors')

# add decision tree classifier
C = 1
kernel = 'linear'
n_neighbors = 5
max_depth = 9

from sklearn import tree
dtree = tree.DecisionTreeClassifier(max_depth=1, random_state=2)
dtree.fit(X_train, y_train) 
score_dtree = dtree.score(X_test, y_test)
#print("dtree_score is " + str(score_dtree))
benchmark(dtree, X_train, X_test, y_train, y_test, 'Dtree')


