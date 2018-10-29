# -*- coding: utf-8 -*-
"""
Use K mean clustering to group the employee's address locations to 10 clusters.
Find the 10 bus stops that are closest to the centre point of the 10 mean cluster 
"""
import pandas as pd
from math import pi, cos
from sklearn.cluster import KMeans, SpectralClustering
from geopy.distance import vincenty
import matplotlib.pyplot as plt
# Importing GPS . lat and long are the 2 features

employees_address= pd.read_csv("Employee_AddressWithGPS.csv")
employees_lat = employees_address['employees_lat'].tolist() #feature 1
employees_long = employees_address['employees_long'].tolist() # feature 2

#I am trying to project the lat & long to XY coordinate first so I can draw it on a 2D plan so i can use it in Kmean to calculate the distance
#Assume the base location (0, 0) is at GPS(37.70, - 122.46)
#R is the radius of the earth, R = 6367 km
R = 6367 
lat0 = 37.70 
long0 = -122.46 

#See this http://mathforum.org/library/drmath/view/51833.html
def XDistance(lat0, long0 , lat1, long1):
    X = R*(long1-long0)*(pi/180)*cos(lat0)
    return X

def yDistance(lat0, long0 , lat1, long1): 
    y = R*(lat1-lat0)*pi/180
    return y

employees_address["employees_X"] = XDistance(lat0, long0 , employees_address['employees_lat'], employees_address['employees_long'])
employees_address["employees_y"] = yDistance(lat0, long0 , employees_address['employees_lat'], employees_address['employees_long'])

employees_X =employees_address['employees_X'].tolist()
employees_y =employees_address['employees_y'].tolist()

plt.scatter(employees_lat,employees_long, marker="x", color='r')
plt.show()

plt.scatter(employees_X,employees_y, marker="x", color='r')
plt.show()

bus_address= pd.read_csv("Potentail_Bust_Stops_withGPS.csv")
coords_bus = bus_address[['bus_lat','bus_long']].values
#print(coords_bus)


# K mean 
# 1. Find 10 inital centre points randomly
# 2. Calculate distance between each of the data with each of the centre point
# 3. Recalculate the centre point by calculating the mean
# 4. Repeat step 3 and 4 untill the old and new centre points are very similar

#Potentially repeat this following step for 3~5 times with different random state value in case if we have a bad initial guess
model =  KMeans(n_clusters=10, random_state=0).fit(employees_address[['employees_X','employees_y']])
labels = model.labels_
centroids = model.cluster_centers_

plt.scatter(centroids[:,0], centroids[:,1], marker="x", color='r')
plt.show()
#print(centroids)

#Conver Centroids back to GPS points from XY coordinates
def GPSlat(lat0, long0 , X, y): 
  
    lat1 = (y / ((pi/180 ) * R)) + lat0
    
    return lat1

def GPSlong(lat0, long0 , X, y):

    long1 = (X / (cos(lat0)*(pi/180)) + (R*long0))/R
    return long1

centroids_lats = []
centroids_longs = []
cens =[]
for centre in centroids:
    centroids_lat = GPSlat(lat0, long0 , centre[0], centre[1])
    centroids_long = GPSlong(lat0, long0 , centre[0], centre[1])
    centroids_lats.append(centroids_lat)
    centroids_longs.append(centroids_long)
    cens.append([centroids_lat,centroids_long])

print(centroids_lats)
print(centroids_longs)
print(cens)
plt.scatter(centroids_lats, centroids_longs, marker="x", color='r')
plt.show()

i = 1

buses =[]


#Customize a disance function
def Distance(centre, busStop):
    #print(vincenty(centre, busStop))
    distance = str(vincenty(centre, busStop))
    return(float(distance.replace(" km", "")))

for centre in cens:
    #Find the bus stop that's closest to the centre
    distance = 100000
    #print("Centroid #" + str(i))
   
    for bus in coords_bus:
        if Distance(centre, bus) < distance:
            distance = Distance(centre, bus)
            bus_nearest = bus
    #print("Nearest Bus Found for cluster " + str(i))        
    #print(bus_nearest)
    i +=1
    buses.append(list(bus_nearest))

print("GPS of 10 buses chosen are")
print(buses)

bus_lats = []
bus_longs = []
for b in buses:
    bus_lats.append(b[0])
    bus_longs.append(b[1])

plt.scatter(bus_lats,bus_longs, marker="x", color='r')
plt.show()