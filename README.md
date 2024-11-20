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
In this section, we explore the relationships between different variables in the dataset. We use Pandas, Seaborn, and Matplotlib libraries for various visualizations to help uncover patterns and gain insights. Below are the key visualizations:

**1. Age vs. 10-Year Heart Disease Risk:** To better understand this relationship, age was binned in 5-year intervals. The analysis revealed that individuals aged 45 to 65 years are the most at risk for heart disease.

**2. Displot: Systolic Blood Pressure vs. 10-Year Heart Disease Risk:** Patients with systolic blood pressure levels above 150 are more likely to develop heart disease within the next 10 years.
   
**3. Violin Plot: BMI vs. Hypertension:** The violin plot shows us that individuals with a BMI rate of 27 or higher are also more likely to have hypertension problems.
   
**4. Stacked Bar Plot: Stroke vs. 10-Year Heart Disease Risk:** Patients with previous strokes are more likely to suffer from heart conditions in ten years.
   

| ![Age_target](https://github.com/juliamartin0/heart_disease/blob/main/age_target.png?raw=true) | ![sysBP_target](https://github.com/juliamartin0/heart_disease/blob/main/sysBP_target.png?raw=true) |
|----------------------------|----------------------------|
| ![bmi_hyp](https://github.com/juliamartin0/heart_disease/blob/main/bmi_hyp.png?raw=true) | ![prevalentStroke_target](https://github.com/juliamartin0/heart_disease/blob/main/stroke_target.png?raw=true) |

For more visualizations and details, please refer to the attached code.

## Logistic Regression Model

Once the data has been cleaned and prepared, and after getting a first look of how the variables act within each other and the target, I applied Logistic Regression to predict the likelihood of heart disease in the next 10 years. We chose Logistic Regression due to its simplicity, efficiency, and the fact that it provides easily interpretable results, making it a strong candidate for binary classification tasks like this one. 

**Model Selection and Variable Testing**
I experimented with different sets of variables to see how they influenced the model's performance, measured by the AUC (Area Under the Curve). Initially, I manually selected a set of variables that seemed intuitively important to predicting heart disease and eliminated those that were highly correlated: 

 - Model: TenYearCHD ~ age + cigsPerDay + sysBP + male + prevalentStroke + diabetes (model 4 image below)
 - AUC: 0.726

<p align="center">
  <img src="https://github.com/juliamartin0/heart_disease/blob/main/models_manual.png?raw=true" alt="manual models"/>
</p> 

In this manual model, I chose a set of variables based on prior knowledge and intuition. It’s worth noting that all the variables in this model were statistically significant, meaning that, based on the model’s results, there is strong evidence to suggest that each of these variables has a meaningful relationship with the outcome variable (heart disease in the next 10 years). In other words, each variable has an influence on the prediction, and their effects are unlikely to have occurred by chance.

While this model gave us a reasonable performance, I wanted to further optimize the selection of variables, so I tried the backwards selection method. This technique iteratively removes variables that don't significantly improve the model's performance, helping to simplify the model while maintaining accuracy.

**Backward Selection Variables**

 - Selected Variables: TenYearCHD ~ 'const', 'age', 'sysBP', 'diaBP', 'glucose', 'prop_missings', 'age_sqr', 'cigsPerDay_raiz4', 'diaBP_sqr', 'BMI_sqr', 'glucose_exp', 'male_1.0', 'currentSmoker_1.0', 'prevalentStroke_1.0', 'prevalentHyp_1.0'
 - AUC: 0.73

The performance of the backward selection model improved slightly to 0.73, but given that the AUC only increased by a small margin, it wasn’t worth adding all the extra complexity (we can see that after 6 variables the AUC score barely increases). In this case, a simpler model with fewer variables still provided comparable performance (see image below), which aligns with the principle of Occam’s Razor — simpler models are often preferred when the performance difference is minimal.

<p align="center">
  <img src="https://github.com/juliamartin0/heart_disease/blob/main/backward_method.png?raw=true" alt="backward selection"/>
</p>

**Interpreting the Model: Odds Ratios**
After fitting the Logistic Regression model, we can interpret the coefficients as odds ratios, which provide insight into the relationship between the predictor variables and the likelihood of having heart disease in the next 10 years. Here's how we interpret the key variables from the simpler model:

 - Prevalent Stroke: A person who has had a prior stroke is almost **3 times more likely** to have heart disease in the next 10 years compared to someone without a prior stroke (Odds Ratio ≈ 3.0).
 - Diabetes: **Diabetic patients are 2.2 times more likely** to develop heart disease compared to non-diabetic patients.
 - Age: For each additional year of age, the odds of having a heart attack increase by 6%. This means that as people get older, their risk of heart disease in the next decade increases.
 - Gender: Males have a 62% higher likelihood of heart disease in 10 years compared to females.

These interpretations hold ceteris paribus (all other variables constant), meaning the effects are measured assuming that the other factors in the model do not change.

## Handling the Unbalanced Dataset

The dataset for predicting heart disease is highly imbalanced, with 85% of individuals not having heart disease and only 15% diagnosed with it. This imbalance can bias the model towards predicting the majority class (no heart disease), leading to a high accuracy but poor performance in detecting heart disease cases.

### Challenge with Default Classification Threshold
The default classification threshold of 0.5 often doesn’t work well in imbalanced datasets. A threshold of 0.5 would predict most people as not having heart disease, but this approach misses many true positive cases (heart disease cases), potentially leading to undetected health risks. While the model would have high accuracy, its ability to correctly identify heart disease cases (sensitivity or recall) would be poor.

### Solution Approach
To address the imbalance and improve model performance, two techniques were applied:

**SMOTE (Synthetic Minority Over-sampling Technique)**
SMOTE was used to generate synthetic examples of the minority class (heart disease cases), making the dataset more balanced. This allows the model to learn better patterns for predicting heart disease, improving its ability to detect heart disease cases.

**Threshold Adjustment for Optimal Sensitivity and Specificity**
I adjusted the classification threshold based on the goal of maximizing true positive detection while minimizing false positives that lead to unnecessary medical follow-ups. Initially, I tried using Youden’s Index, which suggested a threshold of 0.15, but this resulted in too many false positives—the model predicted almost everyone as having heart disease.

<p align="center">
  <img src="https://github.com/juliamartin0/heart_disease/blob/main/youden.png?raw=true" alt="youden"/>
</p>

The threshold of 0.41 was chosen after evaluating different thresholds and measuring the F1-score. It struck the best balance between detecting heart disease cases (true positives) and avoiding excessive false positives (wrongly classifying healthy individuals). This balance is crucial because:

 - Maximizing detection of true positives ensures that people at risk of heart disease are identified and receive timely care.
 - Minimizing false positives avoids unnecessary medical tests, reducing healthcare costs and unnecessary follow-ups.

This updated model, with threshold adjustment and SMOTE to balance the classes, shows a significant improvement in terms of recall for the positive class (heart disease) without sacrificing too much in precision. The key achievements are:

 - A high recall of 0.80 for the positive class (heart disease), indicating that the model is identifying many more positive cases.
 - An F1-score of 0.70 for the positive class, which strikes a good balance, meaning the model is not sacrificing much precision for the high recall.
 - While accuracy remains relatively low (66%), this is expected because improving recall and F1-score often requires focusing on detecting more positive cases (even at the cost of precision). This trade-off is typical when the primary objective is to reduce false negatives.



## Conclusion

The model performs well because it effectively identifies both positive and negative cases of heart disease, without significantly compromising precision. Its main goal is to correctly classify as many people as possible, including those at risk for heart disease (true positives) and those without it (true negatives). Therefore, this approach is highly valid for ensuring both early intervention and resource efficiency in a real-world setting..

When validated with unseen data, the model consistently performs with a standard deviation of ±1.18% across different test sets. This indicates that the **model's performance is stable and reliable**, showing minimal variability (less than 1.2%) across different data splits. In other words, the model is **robust** and **not overfitting**, meaning it's not just memorizing specific data, but generalizing well to new, unseen cases.

In real-world applications like healthcare, where the cost of missing a heart disease case (false negative) can have serious consequences, the goal is to strike a balance between correctly identifying positive cases (heart disease) and negative cases (no heart disease). A good F1 score is key to achieving this balance. By prioritizing F1 score, the model aims to maximize the identification of both true positives and true negatives, minimizing false negatives (which could lead to undiagnosed patients) while avoiding an overload of false positives, which could result in unnecessary follow-ups and treatments. This approach ensures that the model is both accurate and efficient in detecting those at risk, while not overwhelming healthcare resources with false alarms.



