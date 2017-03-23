#5. Data Modeling > Lab: Clustering > Assignment 3 - Telephone Lat Long
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot') # Look Pretty
from sklearn.cluster import KMeans
#
# INFO: This dataset has call records for 10 users tracked over the course of 3 years.
# Your job is to find out where the users likely live at!


def showandtell(title=None):
  if title != None: plt.savefig(title + ".png", bbox_inches='tight', dpi=300)
  plt.show()
  # exit()

def clusterInfo(model):
  print "Cluster Analysis Inertia: ", model.inertia_
  print '------------------------------------------'
  for i in range(len(model.cluster_centers_)):
    print "\n  Cluster ", i
    print "    Centroid ", model.cluster_centers_[i]
    print "    #Samples ", (model.labels_==i).sum() # NumPy Power

# Find the cluster with the least # attached nodes
def clusterWithFewestSamples(model):
  # Ensure there's at least on cluster...
  minSamples = len(model.labels_)
  minCluster = 0
  for i in range(len(model.cluster_centers_)):
    if minSamples > (model.labels_==i).sum():
      minCluster = i
      minSamples = (model.labels_==i).sum()
  print "\n  Cluster With Fewest Samples: ", minCluster
  return (model.labels_==minCluster)


def doKMeans(data, clusters=0):
  #
  # TODO: Be sure to only feed in Lat and Lon coordinates to the KMeans algo, since none of the other
  # data is suitable for your purposes. Since both Lat and Lon are (approximately) on the same scale,
  # no feature scaling is required. Print out the centroid locations and add them onto your scatter
  # plot. Use a distinguishable marker and color.
  #
  # Hint: Make sure you fit ONLY the coordinates, and in the CORRECT order (lat first). This is part
  # of your domain expertise. Also, *YOU* need to instantiate (and return) the variable named `model`
  # here, which will be a SKLearn K-Means model for this to work.
  #
  # 
  model = KMeans(clusters)
  model.fit(data)
  return model



#
# TODO: Load up the dataset and take a peek at its head and dtypes.
# Convert the date using pd.to_datetime, and the time using pd.to_timedelta
#
# 
#import os

df = pd.read_csv('Datasets/CDR.csv')
df.CallDate = pd.to_datetime(df.CallDate, errors='coerce')
df.CallTime = pd.to_timedelta(df.CallTime, errors='coerce')




#
# TODO: Create a unique list of of the phone-number values (users) stored in the
# "In" column of the dataset, and save it to a variable called `unique_numbers`.
# Manually check through unique_numbers to ensure the order the numbers appear is
# the same order they appear (uniquely) in your dataset:
#
# .. your code here ..
unique_numbers = df.In.unique()
'''
Out[219]: 
array([4638472273, 1559410755, 4931532174, 2419930464, 1884182865,
       3688089071, 4555003213, 2068627935, 2894365987, 8549533077], dtype=int64)

type(nIn)
Out[220]: numpy.ndarray
'''

#
# INFO: The locations map above should be too "busy" to really wrap your head around. This
# is where domain expertise comes into play. Your intuition tells you that people are likely
# to behave differently on weekends:
#
# On Weekdays:
#   1. People probably don't go into work
#   2. They probably sleep in late on Saturday
#   3. They probably run a bunch of random errands, since they couldn't during the week
#   4. They should be home, at least during the very late hours, e.g. 1-4 AM
#
# On Weekdays:
#   1. People probably are at work during normal working hours
#   2. They probably are at home in the early morning and during the late night
#   3. They probably spend time commuting between work and home everyday




print "\n\nExamining person: ", 0
# 
# TODO: Create a slice called user1 that filters to only include dataset records where the
# "In" feature (user phone number) is equal to the first number on your unique list above
#
# 
#u = df[df['In'] == 4638472273]
u = df[df['In'] == 4638472273]

#
# TODO: Alter your slice so that it includes only Weekday (Mon-Fri) values.
#
# 
u1 = u[u['DOW'] != 'Sat']
u2 = u1[u1['DOW'] != 'Sun']
'''
u2.DOW.unique()
Out[258]: array(['Mon', 'Tue', 'Wed', 'Thr', 'Fri'], dtype=object)
'''


#
# TODO: The idea is that the call was placed before 5pm. From Midnight-730a, the user is
# probably sleeping and won't call / wake up to take a call. There should be a brief time
# in the morning during their commute to work, then they'll spend the entire day at work.
# So the assumption is that most of the time is spent either at work, or in 2nd, at home.
#
# 
user1 = u2[u2['CallTime'] < '17:00:00']

#
# TODO: Plot the Cell Towers the user connected to
#
# 
fig = plt.figure()
ax = fig.add_subplot(111)

u3 = user1[['TowerLat','TowerLon']]
ax.scatter(u3.TowerLat, u3.TowerLon, c='blue', alpha=0.1)

print(u3.shape)
#
# INFO: Run K-Means with K=3 or K=4. There really should only be a two areas of concentration. If you
# notice multiple areas that are "hot" (multiple areas the usr spends a lot of time at that are FAR
# apart from one another), then increase K=5, with the goal being that all centroids except two will
# sweep up the annoying outliers and not-home, not-work travel occasions. the other two will zero in
# on the user's approximate home location and work locations. Or rather the location of the cell
# tower closest to them.....
model = doKMeans(u3, 4)


#
# INFO: Print out the mean CallTime value for the samples belonging to the cluster with the LEAST
# samples attached to it. If our logic is correct, the cluster with the MOST samples will be work.
# The cluster with the 2nd most samples will be home. And the K=3 cluster with the least samples
# should be somewhere in between the two. What time, on average, is the user in between home and
# work, between the midnight and 5pm?
midWayClusterIndices = clusterWithFewestSamples(model)
midWaySamples = user1[midWayClusterIndices]
print "    Its Waypoint Time: ", midWaySamples.CallTime.mean()


#
# Let's visualize the results!
# First draw the X's for the clusters:
ax.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s=169, c='r', marker='x', alpha=0.8, linewidths=2)
print(model.cluster_centers_)
clusterInfo(model)
#
# Then save the results:
showandtell('Weekday Calls Centroids')  # Comment this line out when you're ready to proceed

'''
# decimal
lat long for post office
#32.7219188
#-96.890399

1559410755: centroid for the cluster with most samples:
    Cluster  0
    Centroid  [ 32.69557708 -96.93522725]
    #Samples  3121

2068627935: 
      Cluster  1
    Centroid  [ 32.72089181 -96.83278954]
    #Samples  1497
  
2894365987
  Cluster  1
    Centroid  [ 32.72165711 -96.89201378]
    #Samples  3288
    
3688089071
  Cluster  2
    Centroid  [ 32.81198486 -96.87034706]
    #Samples  456
'''