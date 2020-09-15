# Rental price estimate for real estate in Switzerland

# Crawling

## Source

Rental objects: https://www.immoscout24.ch

Train stations: https://data.sbb.ch

## Description

The aim of the script is to fetch all rental apartments and houses in Switzerland for private living from Immoscout24. 
The calculation of the distance to the nearest railway station takes place directly after the fetching of a data object.

The data of all stations and the rental objects are transmitted via the respective shadow APIs via JSON, which enables a high-performance query.

In October 2019 I was able to retrieve exactly 38,150 rental objects in 2 hours, which can be checked via a log file. 
The corresponding data is stored in a CSV incl. header. Immoscout24's immoId ensures that the properties are not stored twice.

As the first, the JSON with all stations in Switzerland is fetched from the SBB. 
Subsequently, the URL is titrated through, whereby data objects are always output up to approx. l = 5700 (as of October 2019). 
As an answer one gets a JSON with all rental objects and the partial path, which one must concatenate with the path of the shadow API, 
in order to finally reach all detailed information of a building (also JSON).

It is possible that Immoscout24 has changed its API in the meantime, so that the script has to be adapted accordingly. 

## Attributes

- `immoId` (int): unique identifier of Immoscout24
- `objType` (chr): "house" or "flat"
- `cityName` (chr): name of the city or municipality
- `zipCode` (int): swiss postal code
- `regionId` (int): unique region identifier of Immoscout24
- `canton` (chr): swiss canton
- `street` (chr): street of the rental object
- `rooms` (dbl): number of rooms (e.g. 3.5)
- `floor` (int): floor of the rental object
- `surface` (int): living surface of the rental object in $m^2$
- `yearBuilt` (int): building year
- `yearRenovated` (int): year of renovation if exists
- `lon` (dbl): longitude WGS84
- `lat` (dbl): latitude WGS84
- `distanceToStation` (dbl): distance to the next train station in switherland in km (calculated, not fetched from immoscout24)
- `netPrice` (int): net price in CHF
- `extraPrice` (int): extra price in CHF
- `price` (int): price in CHF (target variable)