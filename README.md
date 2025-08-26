**Crime Type & Location Prediction**<br>
Authors: Ori Fogel, Iris Birman

**Overview**<br>
This project uses data science to predict future crime types and locations, providing insights for proactive crime prevention and optimized resource allocation. 

**Objectives**<br>
Predict high-risk areas and likely crime types.<br>
Enable data-driven crime prevention strategies.<br>
Optimize allocation of resources based on predicted risk.<br>

**Data**<br>
Source: Los Angeles Police Department (LAPD) via data.gov<br>
Includes historical crime records with location, time, and type.<br>

**Methodology**<br>
Data Preprocessing: Cleaning, feature engineering, and handling class imbalance.<br>
Modeling: Random Forest performed best with all relevant features.<br>
Balancing: RandomOverSampler used to address class imbalance.<br>
Evaluation: Precision, recall, and F1-score metrics.<br>

**Dashboard**<br>
Explore the interactive dashboard: [Crime Predictions Dashboard](https://huggingface.co/spaces/orifogel/crime_predictions)

**Outcomes**<br>
The model predicts crime types relatively well, though improvements are possible with additional features or alternative data cleaning.<br>
Predicting geographic area (Bureau) is more challenging, suggesting data limitations.<br>
Future work could explore more localized or specialized machine learning models for better performance.<br>

**Technologies**<br>
Python, Pandas, NumPy, Scikit-learn, Imbalanced-learn, Geopandas, Matplotlib/Seaborn
