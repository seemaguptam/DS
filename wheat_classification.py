# 5. Data Modeling > Lab: K-Nearest Neighbors > Assignment 5 - KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

from sklearn.model_selection import train_test_split

matplotlib.style.use('ggplot') # Look Pretty


def plotDecisionBoundary(model, X, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.6
  resolution = 0.0025
  colors = ['royalblue','forestgreen','ghostwhite']

  # Calculate the   boundaris
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix
  # are the predictions of the class at at said location
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  cs = plt.contourf(xx, yy, Z, cmap=plt.cm.terrain)

  # Plot the test original points as well...
  for label in range(len(np.unique(y))):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], label=str(label), alpha=0.8)

  p = model.get_params()
  plt.axis('tight')
  plt.title('K = ' + str(p['n_neighbors']))


# 
# TODO: Load up the dataset into a variable called X. Check the .head and
# compare it to the file you loaded in a text editor. Make sure you're
# loading your data properly--don't fail on the 1st step!
#

X = pd.read_csv("Datasets/wheat.data")

'''
X.shape
Out[8]: (210, 9)
'''

#
# TODO: Copy the 'wheat_type' series slice out of X, and into a series
# called 'y'. Then drop the original 'wheat_type' column from the X
#
# .. your code here ..
#y = X.drop('wheat_type',axis=1)
yc = X['wheat_type'].copy()
X.drop(labels=['wheat_type'], inplace=True, axis=1)


# TODO: Do a quick, "ordinal" conversion of 'y'. In actuality our
# classification isn't ordinal, but just as an experiment...
#
'''
type(yc)
Out[15]: pandas.core.series.Series
yc.unique()
Out[18]: array(['kama', 'canadian', 'rosa'], dtype=object)
yc.value_counts()
Out[20]: 
canadian    76
rosa        68
kama        66
Name: wheat_type, dtype: int64
yc.shape
Out[28]: (210L,)
y.shape
Out[30]: (210L,)

#df['col3'] = df['col3'].astype('category')
from sklearn import preprocessing
>>> le = preprocessing.LabelEncoder()
>>> le.fit(["paris", "paris", "tokyo", "amsterdam"])
LabelEncoder()
>>> list(le.classes_)
['amsterdam', 'paris', 'tokyo']
>>> le.transform(["tokyo", "tokyo", "paris"]) 
array([2, 2, 1]...)
>>> list(le.inverse_transform([2, 2, 1]))
['tokyo', 'tokyo', 'paris']
'''
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(yc)
y = le.transform(yc)

#
# TODO: Basic nan munging. Fill each row's nans with the mean of the feature
# df.my_feature.fillna( df.my_feature.mean() )
# df.my_feature.unique()
# df.my_feature.value_counts()
# X.isnull().any()
'''
X.shape
Out[32]: (210, 8)
X.isnull().any()
Out[87]: 
id             False
area           False
perimeter      False
compactness     True
length         False
width           True
asymmetry      False
groove          True
dtype: bool
'''
X.compactness = X.compactness.fillna(X.compactness.mean())
X.width       = X.width.fillna(X.width.mean())
X.groove      = X.groove.fillna(X.groove.mean())

'''
X.isnull().any()
Out[96]: 
id             False
area           False
perimeter      False
compactness    False
length         False
width          False
asymmetry      False
groove         False
dtype: bool
'''


#
# TODO: Split X into training and testing data sets using train_test_split().
# INFO: Use 0.33 test size, and use random_state=1. This is important
# so that your answers are verifiable. In the real world, you wouldn't
# specify a random_state.
#
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

'''
X_train.shape
Out[21]: (140, 8)

X_test.shape
Out[22]: (70, 8)

y_train.shape
Out[23]: (140L,)

y_test.shape
Out[24]: (70L,)

X.head(n=5)
Out[25]: 
   id   area  perimeter  compactness  length  width  asymmetry  groove
0   0  15.26      14.84       0.8710   5.763  3.312      2.221   5.220
1   1  14.88      14.57       0.8811   5.554  3.333      1.018   4.956
2   2  14.29      14.09       0.9050   5.291  3.337      2.699   4.825
3   3  13.84      13.94       0.8955   5.324  3.379      2.259   4.805
4   4  16.14      14.99       0.9034   5.658  3.562      1.355   5.175
'''
# 
# TODO: Create an instance of SKLearn's Normalizer class and then train it
# using its .fit() method against your *training* data.
#  
# NOTE: The reason you only fit against your training data is because in a
# real-world situation, you'll only have your training data to train with!
# In this lab setting, you have both train+test data; but in the wild,
# you'll only have your training data, and then unlabeled data you want to
# apply your models to.
#
# 
from sklearn.preprocessing import Normalizer 
scaler = Normalizer().fit(X_train) 


#
# TODO: With your trained pre-processor, transform both your training AND
# testing data.
#
# NOTE: Any testing data has to be transformed with your preprocessor
# that has ben fit against your training data, so that it exist in the same
# feature-space as the original data used to train your models.
#
# .. your code here ..

nX_train = scaler.transform(X_train) 
nX_test = scaler.transform(X_test)
 


#
# TODO: Just like your preprocessing transformation, create a PCA
# transformation as well. Fit it against your training data, and then
# project your training and testing features into PCA space using the
# PCA model's .transform() method.
#
# NOTE: This has to be done because the only way to visualize the decision
# boundary in 2D would be if your KNN algo ran in 2D as well:
'''
from sklearn.decomposition import PCA
#pca = PCA(n_components=2, svd_solver='full')
pca = PCA(n_components=2, random_state=1)
pca.fit(df)
PCA(copy=True, n_components=2, whiten=False)

T = pca.transform(df)

from sklearn.decomposition import PCA 
pca = PCA(n_components=0.95) 
'''
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
#pca = PCA(n_components=2, random_state=1)
pca.fit(nX_train)
#PCA(copy=True, n_components=2, whiten=False)
pnX_train = pca.transform(nX_train)
pnX_test = pca.transform(nX_test)

'''
pnX_train.shape
Out[22]: (140L, 2L)

pnX_test.shape
Out[23]: (70L, 2L)
'''
#
# TODO: Create and train a KNeighborsClassifier. Start with K=9 neighbors.
# NOTE: Be sure train your classifier against the pre-processed, PCA-
# transformed training data above! You do not, of course, need to transform
# your labels.
#
# 
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(pnX_train, y_train) 



# HINT: Ensure your KNeighbors classifier object from earlier is called 'knn'
plotDecisionBoundary(knn, pnX_train, y_train)


#------------------------------------
#
# TODO: Display the accuracy score of your test data/labels, computed by
# your KNeighbors model.
#
# NOTE: You do NOT have to run .predict before calling .score, since
# .score will take care of running your predictions for you automatically.
#
# 
print(knn.score(pnX_test, y_test))

'''
scores with pca
knn9.score(pnX_test, y_test)
Out[25]: 0.88571428571428568
knn8.score(pnX_test, y_test)
Out[29]: 0.90000000000000002
knn6.score(pnX_test, y_test)
Out[32]: 0.91428571428571426
knn4.score(pnX_test, y_test)
Out[36]: 0.90000000000000002
knn2.score(pnX_test, y_test)
Out[39]: 0.90000000000000002
knn1.score(pnX_test, y_test)
Out[42]: 0.90000000000000002
-----
scores without PCA are:
Out[44]: 0.91428571428571426
knn9.score(nX_test, y_test)
Out[44]: 0.91428571428571426
knn8.score(nX_test, y_test)
Out[47]: 0.9285714285714286
knn6.score(nX_test, y_test)
Out[50]: 0.95714285714285718
knn3.score(nX_test, y_test)
Out[53]: 0.97142857142857142
knn2.score(nX_test, y_test)
Out[59]: 0.95714285714285718
knn1.score(nX_test, y_test)
Out[56]: 0.95714285714285718
'''
#
# BONUS: Instead of the ordinal conversion, try and get this assignment
# working with a proper Pandas get_dummies for feature encoding. HINT:
# You might have to update some of the plotDecisionBoundary code.


plt.show()

''' 
The problem: DAT210x 5. Data Modeling > Lab: K-Nearest Neighbors > Assignment 5

'''