# -*- coding: utf-8 -*-

from math import sin, cos, sqrt, atan2, radians
import requests, csv, json, os

na = "<NA>"
objFilePath = 'object_file.csv'
objIds = None

def testEqualityOfJsonResponse(url1, url2):
    #result1 = jsonRequest("https://rest-api.immoscout24.ch/v4/en/properties/5702801?ci=1&ct=64&l=4020&pn=1&pty=1%2C4&s=1&t=1")
    #result2 = jsonRequest("https://rest-api.immoscout24.ch/v4/en/properties/5702801")
    result1 = jsonRequest(url1)
    result2 = jsonRequest(url2)

    if result1 == result2:
        print(result1)
        print("both same")
    else:
        print("different")

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
    with open('Haltestellen-liste.json') as json_file:
        trainStops = json.load(json_file)
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
def launchListCrawling(json):
    global objIds
    objIds = getObjIds()
    matches = json["pagingInfo"]["totalMatches"]
    path = "https://rest-api.immoscout24.ch/v4/en/properties/"
    for i in range(matches):  
        resource = json["allProperties"][i]["detailUrl"]["en"].split("/")[-1]
        header = path+resource
        
        #avoid double requests of the same building
        identificator = resource.split("?")[0]
        if identificator in objIds:
            continue
        
        #perform actual request
        obj = jsonRequest(header)
        if obj:
            objIds[identificator] = True
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
                print("writing index: ", i)
                object_writer = csv.writer(object_file)
                object_writer.writerow([identificator, cityName, zipCode, regionId, canton, street, rooms,
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


            
def getValue(json, key):
    try:
        value = json[key]
    except:
        value = na
    return value
            
    

def storeZip(json, index):
    with open('zip_file.csv', mode='a', newline='') as zip_file:
        zip_writer = csv.writer(zip_file)
        zip_writer.writerow([index, json["properties"][0]["cityName"], json["properties"][0]["zip"]])
    
  

def launchCrawling(zips = False):
    failedCrawlsInRow = 0
    d = {}
    #range until ~5700
    for i in range(2, 3):
        listUrl = "https://rest-api.immoscout24.ch/v4/en/properties?l={}&s=1&t=1".format(i)
        result = jsonRequest(listUrl)
        if result:
            try:
                #request not failed if JSON correct but no objects found
                if result["pagingInfo"]["totalMatches"] > 0:
                    failedCrawlsInRow = 0
                    #creates a file with just the id's of zips if it is set
                    if zips:
                        if result["properties"][0]["zip"] in d:
                            continue
                        d[str(result["properties"][0]["zip"])] = True
                        print("i: ", i, "\ncityZip: ", result["properties"][0]["zip"], "\ncityName: ", result["properties"][0]["cityName"])
                        storeZip(result, i)
                    else:
                        launchListCrawling(result)
                else:
                    failedCrawlsInRow += 1
            except:
                failedCrawlsInRow += 1
        else:
            failedCrawlsInRow += 1
            if failedCrawlsInRow > 20:
                break
        
        
launchCrawling()