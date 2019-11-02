# -*- coding: utf-8 -*-
#!/usr/bin/python3

from math import sin, cos, sqrt, atan2, radians
import requests, csv, os, logging

logging.basicConfig(filename='location.log', format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)
na = "<NA>"
objects = 0
objFilePath = 'object_file.csv'
failedCrawlsInRow = 0
trainStops = None
objIds = None
headers = ["immoId", "objType", "cityName", "zipCode", "regionId", "canton", "street", "rooms",
           "floor", "surface", "yearBuilt", "yearRenovated", "lon", "lat",
           "distanceToStation", "netPrice", "extraPrice", "price"]

def getTrainStations():
    global trainStops
    trainStops = jsonRequest("https://data.sbb.ch/explore/dataset/didok-liste/download/"
                           "?format=json&refine.verkehrsmittel=Zug&refine.betriebspunkttyp=Haltestelle&"
                           "timezone=Europe/Berlin")


def distance(lat1, lon1, lat2, lon2):
    """Calculates the distance of two WGS84 coordinates in km"""
    # approximation of earth radius in km
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

def getDistanceToNearestTrainStation(lon, lat):
    shortestDistance = 100000
    global trainStops
    if not(trainStops):
        getTrainStations()
    for station in trainStops:
        stationLat = station["fields"]["geopos"][0]
        stationLon = station["fields"]["geopos"][1]
        d = distance(lat, lon, stationLat, stationLon)
        if d < shortestDistance:
            shortestDistance = d
    return shortestDistance
            


def jsonRequest(url):
    """Requests the specified API,
    returns JSON if successful"""
    try:
        response = requests.get(url)
        return response.json()
    except:
        return None


#python package erstellen Struktur etc. um auf VM laufen lassen zu können
def launchListCrawling(jsonObj, objType):
    global objIds, objects
    objIds = getObjIds()
    matches = jsonObj["pagingInfo"]["totalMatches"]
    path = "https://rest-api.immoscout24.ch/v4/en/properties/"
    for i in range(matches):  
        resource = jsonObj["allProperties"][i]["detailUrl"]["en"].split("/")[-1]
        header = path+resource
        
        #avoid double requests of the same building
        immoId = resource.split("?")[0]
        if immoId in objIds:
            continue
        
        #perform actual request
        obj = jsonRequest(header)
        if obj:
            objIds[immoId] = True
            propertyDetails = getValue(obj, "propertyDetails")
            if getValue(propertyDetails, "priceUnitLabel") != "month": #ensures that the price is always monthly
                continue
            #start of value extraciton
            cityName = getValue(propertyDetails, "cityName")
            zipCode = getValue(propertyDetails, "zip")
            regionId = getValue(propertyDetails, "regionId")
            canton = getValue(propertyDetails, "stateShort")
            street = getValue(propertyDetails, "street")
            rooms = getValue(propertyDetails, "numberOfRooms")
            floor = getValue(getValue(propertyDetails, "attributesSize"), "floor")
            surface = getValue(propertyDetails, "surfaceLiving")
            yearBuilt = getValue(getValue(propertyDetails, "attributes"), "yearBuilt")
            yearRenovated = getValue(getValue(propertyDetails, "attributes"), "yearRenovated")
            lon = getValue(propertyDetails, "longitude")
            lat = getValue(propertyDetails, "latitude")
            distanceToStation = na
            if not(lat == na or lon == na):
                distanceToStation = getDistanceToNearestTrainStation(lon, lat)
            netPrice = getValue(propertyDetails, "netPrice")
            extraPrice = getValue(propertyDetails, "extraPrice")
            price = getValue(propertyDetails, "price")
            with open(objFilePath , mode='a', newline='') as object_file:
                object_writer = csv.writer(object_file)
                if os.stat(objFilePath).st_size == 0: #wirte headers if file is empty
                    object_writer.writerow(headers)
                objects += 1
                object_writer.writerow([immoId, objType, cityName, zipCode, regionId, canton, street, rooms,
                                        floor, surface, yearBuilt, yearRenovated, lon, lat,  
                                        distanceToStation, netPrice, extraPrice, price])
                    
            
def getObjIds():
    global objIds
    if objIds:
        return objIds
    elif not(os.path.exists(objFilePath)):
        objIds = {}
    else:
        objIds = {}
        with open (objFilePath, mode='r') as objFile:
            reader = csv.reader(objFile)
            for row in reader:
                try:
                    objIds[row[0]] = True
                except:
                    continue
    return objIds


            
def getValue(jsonObj, key):
    if jsonObj == na:
        return na
    try:
        return jsonObj[key]
    except KeyError:
        return na
    except:
        return "ERROR"
            
    
        
def listRequest(jsonObj, objType):
    global failedCrawlsInRow
    if jsonObj:
        try:
            #request not failed if JSON correct but no objects found
            if jsonObj["pagingInfo"]["totalMatches"] > 0:
                failedCrawlsInRow = 0
                launchListCrawling(jsonObj, objType)
            else:
                failedCrawlsInRow += 1
        except:
            failedCrawlsInRow += 1
    else:
        failedCrawlsInRow += 1
            

def launchCrawling():
    #range until ~5700 (10.2019)
    for i in range(2, 5700):
        global objects
        objects = 0
        if failedCrawlsInRow > 100:
            break
        houseUrl = "https://rest-api.immoscout24.ch/v4/en/properties?l={}&s=3&t=1".format(i) #s=3 house
        flatUrl = "https://rest-api.immoscout24.ch/v4/en/properties?l={}&s=2&t=1".format(i) #s=2 flat
        listRequest(jsonRequest(houseUrl), "house")
        listRequest(jsonRequest(flatUrl), "flat")
        finished = "objects: {},    finished l={}".format(objects, i)
        logging.info(finished)
        print(finished)
        
        
if __name__ == '__main__':        
    launchCrawling()

