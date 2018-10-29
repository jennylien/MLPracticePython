# -*- coding: utf-8 -*-
"""
Find GPS points for employees using the address.csv given
"""
import pandas as pd
from geopy.geocoders import Nominatim
#https://geopy.readthedocs.io/en/stable/

#link to the mapping search engine tool
geolocator = Nominatim(user_agent="jenny")
"""
Many of the bus addresses returns to None GPS values using Nominatim tool....If this is a commercial product it's better to use Google Map's API
Google map does a good job to find approx. location so it was used to find many of the buses GPS location instead...
"""
#load in csv and creat dataframe for the employee address data
employees_data = pd.read_csv("Employee_Addresses.csv")
employees_list = employees_data["address"].tolist()
employees_lat = []
employees_long = []

for i in employees_list:
    try:
        location = geolocator.geocode(i)
        employees_lat.append(location.latitude) 
        employees_long.append(location.longitude) 
    except:
        employees_lat.append(0)
        employees_long.append(0)
        continue

    
print(employees_lat)
print(employees_long)

employees_lat = pd.Series(employees_lat)
employees_long = pd.Series(employees_long)
employees_data['employees_lat'] = employees_lat.values
employees_data['employees_long'] = employees_long.values

employees_data = employees_data[employees_data['employees_lat'] != 0]
employees_data = employees_data[employees_data['employees_long'] != 0]
employees_data.to_csv("Employee_AddressWithGPS.csv")


