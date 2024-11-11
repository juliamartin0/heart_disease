### 10-Year Heart Disease Prediction Using Logistic Regression

The World Health Organization has estimated 12 million deaths occur worldwide every year due to heart diseases. Half of the deaths in the US and other developed countries are due to cardiovascular diseases, and early detection plays a critical role in preventing adverse outcomes. This repository uses clinical data and machine learning techniques to predict the likelihood of developing heart disease over a 10-year period. By leveraging logistic regression, a commonly used classification algorithm, we aim to identify high-risk individuals who may benefit from early intervention, lifestyle changes, or closer medical monitoring. 

Insights and recommendations are provided on the following key areas:

- **Heart Disease Prediction:** The goal is to build a model that predicts the likelihood of developing heart disease based on different medical and individual factors such as age, cholesterol levels, blood pressure, smoking, and more.
- **Identifying Key Risk Factors:** By analyzing the data, we identify the most significant factors that contribute to the risk of heart disease, which can help in early diagnosis and prevention.
- **Providing Healthcare Insights:** The predictions from the model could assist healthcare providers in identifying high-risk individuals and offering them early intervention, lifestyle changes, or regular monitoring.

### Dataset Overview

This repository uses a dataset that is publicly available on Kaggle, and it is from an ongoing cardiovascular study on residents of the town of Framingham in Massachusetts. This dataset contains over 4,000 patient records and 15 attributes including the target variable, which is whether or not the individual will develop heart disease within 10 years (binary classification: 0 = no, 1 = yes). Each of these attributes is a potential risk factor for future cardiovascular disease. Some of these include age, total cholesterol, prevalent heart strokes, or diabetes. For a full list of features and more detailed descriptions, you can view the dataset on Kaggle.

# Data Structure & Initial Checks

This dataset contained 15 variables: 7 nominal and 9 continuous. However, there is one attribute 'Education', that has no description, so I decided to eliminate it from my study to avoid confusion or wrong interpretations. All 7 nominall variables only have 2 values 1=yes and 0=no, so they can be changed into boolean attributes.  



### Data Cleaning and Preprocessing:

The data used for this project needed cleaning and preparation to ensure it was suitable for analysis.
