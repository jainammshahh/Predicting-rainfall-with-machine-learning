# Project Overview

The agricultural industry heavily relies on weather conditions for effective decision-making in farming activities. Predicting weather conditions can be challenging, especially in regions with unpredictable weather patterns, leading to farmers making suboptimal decisions that impact yields and cost them valuable resources. 

Traditional methods of weather forecasting can be time-consuming and may not always provide accurate results. In recent years, climate change has made weather patterns even more unpredictable, exacerbating the problem of inaccurate weather predictions. There is a pressing need to develop more accurate and efficient methods for predicting rainfall.

Machine learning algorithms have shown promise in weather prediction, as they can process large amounts of data and identify complex patterns that traditional methods may miss. Developing an accurate machine learning algorithm for rainfall prediction will be beneficial for the agricultural industry, especially for farmers in regions with unpredictable weather patterns.

# About Data

Introduction and Description of Data

The unpredictability of weather patterns has been a significant challenge.This project aims to provide a practical solution to address the challenges of weather prediction in the agricultural industry, contributing to the overall growth and sustainability of the sector.

The data has been conveniently split into train and test datasets. In each train and test, weather data is present which consists of locations named region A through region E, which are all neighbouring regions including an anonymized date column. The further features included in the dataset are as follows:

<img width="452" alt="Picture1" src="https://github.com/jainammshahh/Predicting-rainfall-with-machine-learning/assets/114266749/c411c216-e4ef-4e67-9bad-0547c26931ef">

As it is seen from the dataset, the “date” column is anonymized to some random values. There are in total 10 features in the dataset which consists of types and abbreviations of  temperature, wind speed, precipitation, wind speed direction and atmospheric pressure.

The dataset contains 5 csv files in each training and testing set along with a separate csv file named “solution_format.csv” containing target rain predictions for each of the dates, which allows us to use supervised learning when building the model.


The goal currently is to predict the weather for the next day based on three labels:  
●	N - No rain  
●	L - Light Rain  
●	H - Heavy Rain  
