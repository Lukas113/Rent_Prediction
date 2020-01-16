# ImmoRent: Rental price estimate for real estate in Switzerland

Studierender: Lukas Gehrig  
Fachcoach: Michael Graber  

# Overview

## Portfolio-Credit Distribution

- Web Datenbeschaffung: 1
- Explorative Datenanalyse: 1
- Supervised Learning: 2

## Web Data Retrieval

Details regarding the web data retrieval can be found in the folder 'crawling'. In 'crawling.py' you will find the script that retrieves the data, which is stored in the folder 'data_file' under 'object_file - original.csv'.
The script no longer works at this point in time, as I assume that ImmoScout24 has changed something in your JSON structure. More Details are described in the README.md in the 'crawling' folder.

## Data Analysis

The main part of the data analysis was executed in the Jupyter Notebook 'analysis.ipynb' in the folder 'data_analysis'. The analysis of the geographical distribution is located at the end of the notebook 'linear_regression.ipynb' in the folder 'linear_regression'.
Details are available in README.md in the folder 'data_analysis'.

## Supervised Learning

The Supervised Learning part is divided into two parts, 1. a linear regression and 2. a creation of a neural network.
The Linear Regression can be found in the folder 'linear_regression' in the file 'linear_regression.ipynb'.
The Neural Networks part is in the folder 'NN' in the file 'NeuralNetwork.py'. Unfortunately I could not finish this part completely, because I could not debug the gradient check successfully. For this reason training and performance comparison between both models is missing.
More about this can be found in the respective README.md in the corresponding folder.
