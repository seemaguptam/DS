# 5. Data Modeling > Lab: K-Nearest Neighbors > Assignment 5
# If you'd like to try this lab with PCA instead of Isomap,
# as the dimensionality reduction technique:
Test_PCA = True


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  #(2 for benign, 4 for malignant)
  colors = {2:'royalblue',4:'lightsalmon'} 

  
  # Calculate the boundaris
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
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=plt.cm.seismic)
  plt.axis('tight')

  # Plot your testing points as well...
  for label in np.unique(y):
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=colors[label], alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset, identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
# 

X = pd.read_csv("Datasets/breast-cancer-wisconsin.data", names=['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'])
#my_dataframe.columns = ['new', 'column', 'header', 'labels']
#df = df.dropna(axis=0)

'''
1. Sample code number: id number 
2. Clump Thickness: 1 - 10 
3. Uniformity of Cell Size: 1 - 10 
4. Uniformity of Cell Shape: 1 - 10 
5. Marginal Adhesion: 1 - 10 
6. Single Epithelial Cell Size: 1 - 10 
7. Bare Nuclei: 1 - 10 
8. Bland Chromatin: 1 - 10 
9. Normal Nucleoli: 1 - 10 
10. Mitoses: 1 - 10 
11. Class: (2 for benign, 4 for malignant)
df.shape
Out[100]: (699, 11)
df.isnull().any()
Out[102]: 
sample        False
thickness     False
size          False
shape         False
adhesion      False
epithelial    False
nuclei        False
chromatin     False
nucleoli      False
mitoses       False
status        False
dtype: bool
df.head(5)
Out[101]: 
    sample  thickness  size  shape  adhesion  epithelial nuclei  chromatin  \
0  1000025          5     1      1         1           2      1          3   
1  1002945          5     4      4         5           7     10          3   
2  1015425          3     1      1         1           2      2          3   
3  1016277          6     8      8         1           3      4          3   
4  1017023          4     1      1         3           2      1          3   

   nucleoli  mitoses  status  
0         1        1       2  
1         2        1       2  
2         1        1       2  
3         7        1       2  
4         1        1       2  
df.status.unique()
Out[118]: array([2, 4], dtype=int64)

df.status.value_counts()
Out[119]: 
2    458
4    241
Name: status, dtype: int64
df.dtypes
Out[124]: 
sample         int64
thickness      int64
size           int64
shape          int64
adhesion       int64
epithelial     int64
nuclei        object
chromatin      int64
nucleoli       int64
mitoses        int64
status         int64
dtype: object
'''
X.nuclei = pd.to_numeric(X.nuclei, errors='coerce')

'''
X.isnull().any()
Out[128]: 
sample        False
thickness     False
size          False
shape         False
adhesion      False
epithelial    False
nuclei         True
chromatin     False
nucleoli      False
mitoses       False
status        False
dtype: bool
'''
X.nuclei = X.nuclei.fillna(X.nuclei.mean())
#X = X.dropna(axis=0)
'''
X.shape
Out[136]: (699, 11)

X.isnull().any()
Out[137]: 
sample        False
thickness     False
size          False
shape         False
adhesion      False
epithelial    False
nuclei        False
chromatin     False
nucleoli      False
mitoses       False
status        False
dtype: bool
'''
# 
# TODO: Copy out the status column into a slice, then drop it from the main
# dataframe. Always verify you properly executed the drop by double checking
# (printing out the resulting operating)! Many people forget to set the right
# axis here.
#
# If you goofed up on loading the dataset and notice you have a `sample` column,
# this would be a good place to drop that too if you haven't already.
#
# .. your code here ..
#X.status.fillna(X.status.mean())
y = X['status'].copy()
X.drop(labels=['status','sample'], inplace=True, axis=1)



#
# TODO: With the labels safely extracted from the dataset, replace any nan values
# with the mean feature / column value
#
# 




#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in
# the reading material, but set the random_state=7 for reproduceability, and keep
# the test_size at 0.5 (50%).
#
# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=7)




#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that
# the features consist of different units mixed in together, so it might be
# reasonable to assume feature scaling is necessary. Print out a description
# of the dataset, post transformation. Recall: when you do pre-processing,
# which portion of the dataset is your model trained upon? Also which portion(s)
# of your dataset actually get transformed?
#
# 
#from sklearn.preprocessing import Normalizer 
from sklearn.preprocessing import MinMaxScaler 
#from sklearn.preprocessing import RobustScaler 
#from sklearn.preprocessing import StandardScaler
scaler = MinMaxScaler().fit(X_train) 

nX_train = scaler.transform(X_train) 
nX_test = scaler.transform(X_test)
 
'''
k=9 weights=uniform 
After dropping nan nuclei rows
Normalizer score = 0.874269005848
MinMaxScaler score = 0.976608187135
RobustScaler score = 0.973684210526
StandardScaler score = 0.979532163743

After updaing nuclei nan with mean
Normalizer score = 0.842857142857
MinMaxScaler score = 0.965714285714
RobustScaler score = 0.962857142857
StandardScaler score = 0.957142857143

k=9 weights = distance
MinMaxScaler score = 0.965714285714

k=5
MinMaxScaler score = 0.965714285714 for uniform, and 0.96 for distance


'''

#
# PCA and Isomap are your new best friends
model = None
'''if Test_PCA:
  print "Computing 2D Principle Components"
  #
  # TODO: Implement PCA here. Save your model into the variable 'model'.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..

  

else:
  print "Computing 2D Isomap Manifold"
  #
  # TODO: Implement Isomap here. Save your model into the variable 'model'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
'''
# just do PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2, svd_solver='full')
#pca = PCA(n_components=2, random_state=1)
pca.fit(nX_train)
#PCA(copy=True, n_components=2, whiten=False)
pnX_train = pca.transform(nX_train)
pnX_test = pca.transform(nX_test)

  



#
# TODO: Train your model against data_train, then transform both
# data_train and data_test using your model. You can save the results right
# back into the variables themselves.
#
# 



# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D
# training data here. You can use any K value from 1 - 15, so play around
# with it and see what results you can come up. Your goal is to find a
# good balance where you aren't too specific (low-K), nor are you too
# general (high-K). You should also experiment with how changing the weights
# parameter affects the results.
#
# 
from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(pnX_train, y_train) 



#
# INFO: Be sure to always keep the domain of the problem in mind! It's
# WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing
# it to be benign, and then having the patient progress in cancer. Since the UDF
# weights don't give you any class information, the only way to introduce this
# data into SKLearn's KNN Classifier is by "baking" it into your data. For
# example, randomly reducing the ratio of benign samples compared to malignant
# samples from the training set.

# Learning: data here is skewed - there are more instances of benign than of malignant

#
# TODO: Calculate + Print the accuracy of the testing set
#
# 
print(knn.score(pnX_test, y_test))

plotDecisionBoundary(knn, pnX_test, y_test)
