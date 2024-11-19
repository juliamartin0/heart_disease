# 10-Year Heart Disease Prediction Using Logistic Regression

The World Health Organization has estimated 12 million deaths occur worldwide every year due to heart diseases. Half of the deaths in the US and other developed countries are due to cardiovascular diseases, and early detection plays a critical role in preventing adverse outcomes. This repository uses clinical data and machine learning techniques to predict the likelihood of developing heart disease over a 10-year period in adult patients (30-70-year-olds). By leveraging logistic regression, a commonly used classification algorithm, we aim to identify high-risk individuals who may benefit from early intervention, lifestyle changes, or closer medical monitoring. 

Insights and recommendations are provided on the following key areas:

- **Heart Disease Prediction:** The goal is to build a model that predicts the likelihood of developing heart disease based on different medical and individual factors such as age, cholesterol levels, blood pressure, smoking, and more.
- **Identifying Key Risk Factors:** By analyzing the data, we identify the most significant factors that contribute to the risk of heart disease, which can help in early diagnosis and prevention.
- **Providing Healthcare Insights:** The predictions from the model could assist healthcare providers in identifying high-risk individuals and offering them early intervention, lifestyle changes, or regular monitoring.

## Dataset Overview

This repository uses a dataset that is publicly available on Kaggle, and it is from an ongoing cardiovascular study on residents of the town of Framingham in Massachusetts. This dataset contains over 4,000 patient records and 15 attributes including the target variable, which is whether or not the individual will develop heart disease within 10 years (binary classification: 0 = no, 1 = yes). Each of these attributes is a potential risk factor for future cardiovascular disease. Some of these include age, total cholesterol, prevalent heart strokes, or diabetes. For a full list of features and more detailed descriptions, you can view the dataset on Kaggle.

### Data Structure & Initial Checks

This dataset contained 15 variables: 7 nominal and 9 continuous. However, one attribute, **Education**, had no description in the dataset and was therefore removed to avoid confusion or incorrect interpretations. 

- **Nominal Variables:** All 7 nominal variables have only two possible values: 1 (yes) or 0 (no). These were converted into boolean attributes for easier analysis (diabetes, male, prevalentHyp, prevalentStroke, currentSmoker, BPMeds, and TenYearCHD)
- **Continous Variables:** age, cigsPerDay, totChol, sysBP, diaBP, BMI, heartRate, and glucose levels.

After an initial inspection of the data, we observed a few issues that needed attention:

- **Class Imbalance:** The target variable is notably imbalanced, with 85% of individuals labeled as “no” (they will not develop heart disease within 10 years) and only 15% labeled as “yes” (they will develop heart disease). This imbalance may affect the model's ability to accurately predict the minority class ("yes") and requires careful consideration of evaluation metrics and possibly techniques to address imbalance (e.g., resampling or class weighting).
- **Missing Values:** Some attributes such as totChol, BMI heartRate BPMedsor glucose had missing data, which needed to be addressed before modeling.
- **Wide Disparity in Range:** Several continuous variables exhibited a large disparity between the minimum and maximum values, which raised concerns about potential outliers. For example, some individuals had extremely high cholesterol levels or glucose readings, suggesting that further checks were necessary to ensure the data's validity. Such outliers could significantly affect the performance of the predictive model, so these needed to be handled properly during preprocessing.

### Data Cleaning and Preprocessing:

The data used for this project needed cleaning and preparation to ensure it was suitable for analysis.

**1. Outlier Detection:** Outliers, or extreme values, can skew the data and potentially mislead the model. After examining the data, only two variables showed minor skew (leaning towards one side of the range), while glucose was more skewed, with a 6% tilt toward unusually high values. This meant that some individuals had extremely high glucose levels that were not typical for most of the population. To handle these outliers without losing valuable data, I used a method called Winsorization. Winsorization limits the extreme values in each direction by "pulling" them closer to the typical range, which reduces the influence of outliers while keeping all data points in the dataset. This way, the model can train on data that better represents the majority while still accounting for high glucose values in a balanced way.

**2. Handling Missing Values:** 
- I first calculated the percentage of missing values per variable, with the highest being glucose at 10%. Dropping rows with missing data would have reduced the dataset by 12%, so I opted to impute instead given the fact that our dataset is not particularly big.
- Using a missing values matrix (see image), I confirmed that missing data appeared random, there is not a clear pattern of missing values.

![Missing Data Matrix](https://github.com/juliamartin0/heart_disease/blob/main/missing_matrix.png?raw=true)

- **Imputation Strategy:**
  - For nominal variables (those with binary yes/no responses), I used **K-Nearest Neighbors (KNN) imputation**. This method estimates the missing values based on the most similar (or “nearest”) individuals in the dataset, ensuring that the imputed values align with observed data patterns.
  - For categorical variables, I used **mode imputation** (replacing missing values with the most common value in the column). This approach worked well, as categorical variables in this dataset tended to have one dominant value.
 
**3. Correlation Analysis:** High correlation between independent variables can lead to multicollinearity, where the predictor variables are highly related to each other. If the correlation is too high, one or more of the correlated variables may need to be removed to improve the model's performance and reduce overfitting. During the analysis, we observed several pairs of variables with high correlation, which provided important insights:

 - CigsPerDay - currentSmoker: both features are related to smoking habits; someone who smokes a certain numer of cigarettes per day is likely a current smoker.
 - sysBP - diaBP: Systolic blood pressure (sysBP) and diastolic blood pressure (diaBP) are both measurements of blood pressure, and they are naturally correlated.
 - sysBP - prevalentHyp: Prevalence of hypertension is likely to be associated with both systolic and diastolic blood pressure, as high blood pressure is a key feature of hypertension.
 - diaBP - prevalentHyp: similar to sysBP
 - glucose - diabetes: There is a moderate correlation (44%) between glucose levels and diabetes status, as high glucose levels are a key indicator of diabetes. This correlation suggests that these variables are related but not overly redundant.

 ![Correlation Matrix](https://github.com/juliamartin0/heart_disease/blob/main/correlation_matrix.png?raw=true)  

- Cramér's V Analysis: To understand the strength of association between each predictor and the 10-year heart disease risk, we used Cramér's V. This helped us identify which variables are most closely linked to heart disease and should be prioritized in our model. The variables with a stronger association are age, systolic blood pressure, and hypertension, indicating a strong relationship with heart disease risk. This analysis can guide our feature selection and ensure we focus on variables with the most impact on predicting heart disease.

 ![Cramer's V](https://github.com/juliamartin0/heart_disease/blob/main/newplot%20(6).png?raw=true) 

## Exploratory Data Analysis
In this section, we explore the relationships between different variables in the dataset. We use various visualizations to help uncover patterns and gain insights. The full detailed visualizations can be accessed in the code, but here are the main insights: 

| ![Age_target](https://github.com/juliamartin0/heart_disease/blob/main/age_target.png?raw=true) | ![prevalentHyp_target](https://github.com/juliamartin0/heart_disease/blob/main/prevalentHyp_target.png?raw=true) |
|----------------------------|----------------------------|
| ![bmi_hyp](https://github.com/juliamartin0/heart_disease/blob/main/bmi_hyp.png?raw=true) | ![sysBP_target](https://github.com/juliamartin0/heart_disease/blob/main/sysBP_target.png?raw=true) |












