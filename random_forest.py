# the problem: 6. Data Modeling II > Lab: Random Forest > Assignment 6
import pandas as pd
import time

# Grab the DLA HAR dataset from:
# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/static/har/dataset-har-PUC-Rio-ugulino.zip


#
# TODO: Load up the dataset into dataframe 'X'
#
#
import os
#os.chdir("Documents/DS/Dat210x/")
X = pd.read_csv("Datasets/rf.csv")

'''
X.shape
Out[8]: (165633, 19)
X.isnull().any().any()
Out[10]: False
X.gender.unique()
Out[13]: array(['Woman', 'Man'], dtype=object)
X.dtypes
Out[11]: 
user                  object
gender                object
<snip>
'''

#
# TODO: Encode the gender column, 0 as male, 1 as female
# Xi.Private = Xi.Private.map({'Yes':1, 'No':0})
# 
X.gender = X.gender.map({'Man':0, 'Woman':1})
'''
X.dtypes
Out[15]: 
user                  object
gender                 int64
<snip>
 
'''

#
# TODO: Clean up any column with commas in it
# so that they're properly represented as decimals instead
# <<updated data in execl, not required>>
# 


#
# INFO: Check data types
print X.dtypes

'''
X.dtypes
Out[44]: 
user                   object
gender                  int64
age                     int64
how_tall_in_meters    float64
weight                  int64
body_mass_index       float64
x1                      int64
y1                      int64
z1                      int64
x2                      int64
y2                      int64
z2                      int64
x3                      int64
y3                      int64
z3                      int64
x4                      int64
y4                      int64
z4                     object
class                  object
dtype: object
'''
X.z4 = pd.to_numeric(X.z4, errors='coerce')

#
# TODO: Convert any column that needs to be converted into numeric
# use errors='raise'. This will alert you if something ends up being
# problematic
#
# 
X=X.dropna(axis=0)
'''
X.shape
Out[51]: (165632, 19)
X.isnull().any().any()
Out[53]: False
'''

#
# INFO: If you find any problematic records, drop them before calling the
# to_numeric methods above...


#
# TODO: Encode your 'y' value as a dummies version of your dataset's "class" column
#
# 
X = pd.get_dummies(X,columns=['class'])

'''
X.shape
Out[55]: (165632, 23)

X.head(2)
Out[56]: 
     user  gender  age  how_tall_in_meters  weight  body_mass_index  x1  y1  \
0  debora       1   46                1.62      75             28.6  -3  92   
1  debora       1   46                1.62      75             28.6  -3  94   

   z1  x2      ...         y3  z3   x4   y4     z4  class_sitting  \
0 -63 -23      ...        104 -92 -150 -103 -147.0            1.0   
1 -64 -21      ...        104 -90 -149 -104 -145.0            1.0   

   class_sittingdown  class_standing  class_standingup  class_walking  
0                0.0             0.0               0.0            0.0  
1                0.0             0.0               0.0            0.0  

[2 rows x 23 columns]
'''

#
# TODO: Get rid of the user and class columns
#
# .. your code here ..
X.drop(labels=['user'], inplace=True, axis=1)
y = X[['class_sitting', 'class_sittingdown', 'class_standing', 'class_standingup', 'class_walking']]
X.drop(labels=['class_sitting', 'class_sittingdown', 'class_standing', 'class_standingup', 'class_walking'], inplace=True, axis=1)
print X.describe()


#
# INFO: An easy way to show which rows have nans in them
print X[pd.isnull(X).any(axis=1)]



#
# TODO: Create an RForest classifier 'model' and set n_estimators=30,
# the max_depth to 10, and oob_score=True, and random_state=0
'''
>>> from sklearn.ensemble import RandomForestClassifier
>>> model = RandomForestClassifier(n_estimators=10, oob_score=True)
>>> model.fit(X, y)
>>> print model.oob_score_

'''
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=30, oob_score=True, max_depth=10, random_state=0)



# 
# TODO: Split your data into test / train sets
# Your test size can be 30% with random_state 7
# Use variable names: X_train, X_test, y_train, y_test
#
# 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=7)




print "Fitting..."
s = time.time()
#
# TODO: train your model on your training set
#
# .. your code here ..
model.fit(X_train, y_train)
print "Fitting completed in: ", time.time() - s


#
# INFO: Display the OOB Score of your data
score = model.oob_score_
print "OOB Score: ", round(score*100, 3)




print "Scoring..."
s = time.time()
#
# TODO: score your model on your test set
#
# 
s2 = model.score(X_test, y_test)
print "model.score: ", round(s2*100, 3)
print "Scoring completed in: ", time.time() - s


#
# TODO: Answer the lab questions, then come back to experiment more


#
# TODO: Try playing around with the gender column
# Encode it as Male:1, Female:0
# Try encoding it to pandas dummies
# Also try dropping it. See how it affects the score
# This will be a key on how features affect your overall scoring
# and why it's important to choose good ones.



#
# TODO: After that, try messing with 'y'. Right now its encoded with
# dummies try other encoding methods to experiment with the effect.

