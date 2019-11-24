# ImmoRent: Rental price estimate for real estate in Switzerland

Studierender: Lukas Gehrig  
Fachcoach: Michael Graber  

# Data Analysis

In this section, you need to analyze and understand the data you collect to decide which features and data objects to use to create the models.

The first step is to analyze the missing values in the data set. The values from the renovation year are not the same as the other missing values, as there are some houses that have never been renovated before. The missing values are examined absolutely and relatively. In addition, correlations between the missing values are searched for. In addition, the data objects themselves are also examined for the number and distribution of missing values.

The next step is a univariate analysis of the data set to identify the distributions and safe outliers.

The data set is then examined for correlations between the target variable and the remaining features. In addition, I created the 'price per square meter' feature to test the correlation between the features and the two Pries variables. Based on the univariate analysis, I decided to alter the feature 4 digit postal code to a 3- and 2 digit code.

The final step is a final cleanup of the data set. This includes the deletion of some features as well as the discarding of some data objects with too many missing values. More information can be found in the notebook 'analysis.ipynb'.

A total of 37,418 data objects were kept for further processing. This dataset contains the following features:


## Attributes

- `obj_type` (chr): "house" or "flat"
- `rooms` (dbl): number of rooms
- `surface` (int): living surface of the rental object in $m^2$
- `zip_code_2_digits` (int): regional information provided by the first two digits of the postal code
- `zip_code_3_digits` (int): regional information provided by the first three digits of the postal code
- `canton` (chr): swiss canton
- `year_built` (int): building year
- `year_renovated` (int): year of renovation if exists
- `distance_to_station` (dbl): distance to the next train station in switherland in km (calculated, not fetched from immoscout24)
- `price_square_metres` (dbl): calculated price per square metres in CHF (`price` / `surface`)
- `price` (int): price in CHF (target variable)