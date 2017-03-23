# The problem: 5. Data Modeling > Lab: Clustering > Assignment 1
# Import whatever needs to be imported to make this work
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

print("Clustering example")

""">>> kmeans = KMeans(n_clusters=5)
>>> kmeans.fit(df)
KMeans(copy_x=True, init='k-means++', max_iter=300, n_clusters=5, n_init=10,
    n_jobs=1, precompute_distances='auto', random_state=None, tol=0.0001,
    verbose=0)

>>> labels = kmeans.predict(df)
>>> centroids = kmeans.cluster_centers_
"""


# Look Pretty
matplotlib.style.use('ggplot')
plt.style.use('ggplot')


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'



def doKMeans(df):
  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Longitude,
  # and Latitude locations in your dataset. Longitude = x, Latitude = y
  #
  # TODO: Filter df so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #
  df2 = df[['Latitude','Longitude']]
  print("clustering data shape is " + str(df2.shape))
  fig = plt.figure()
  ax = fig.add_subplot(111)
  #ax.scatter(df2.Longitude, df2.Latitude, marker='.', alpha=0.3)
  #
  # TODO: Use K-Means to try and find seven cluster centers in this df.
  # Be sure to name your kmeans model `model` so that the printing works.
  #
  model = KMeans(n_clusters=7)
  model.fit(df2)

  #
  # INFO: Print and plot the centroids...
  centroids = model.cluster_centers_
  
  
  
  #ax.scatter(centroids[:,0], centroids[:,1] , marker='x', c='red')
  ax.scatter(df2.Latitude,df2.Longitude, marker='.', c='grey')
  ax.scatter(centroids[:,0], centroids[:,1], marker='x', c='red', alpha=0.7, linewidths=3, s=169)
  print centroids



#
# TODO: Load your dataset after importing Pandas

df = pd.read_csv("Datasets/GamblingCrimes2001_to_present.csv")
df = df.dropna(axis=0)

#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
df.Date = pd.to_datetime(df.Date, errors='coerce')
print(df.dtypes)



# INFO: Print & Plot your data
doKMeans(df)


#
# TODO: Filter out the data so that it only contains samples that have
# a Date > '2011-01-01', using indexing. Then, in a new figure, plot the
# crime incidents, as well as a new K-Means run's centroids.
# df[ (df.recency < 7) & (df.newbie == 0) ]
#d4=d1[(d1.Date > '2011-01-01')]
d1 = df[['Date','Latitude','Longitude']]
d2 = d1[(d1.Date > '2011-01-01')]





# INFO: Print & Plot your data
doKMeans(d2)
plt.show()


