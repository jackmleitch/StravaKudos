# Strava Kudos Predictor: Project Overview
## Project Overview
* Created a tool that **predicts kudos** (a proxy for user interaction) on [Strava activities](https://www.strava.com/athletes/5028644) (RMSE: 9.41) to see if it was random or if different attributes impact kudos in different ways.
* Attained over 4000 Strava activities using the **Strava API** and python.
* **Engineered new features** using domain knowledge. For example, features encapsulating different run types and times of day were added.
* Performed **feature selection** using a combination of SHAP values and feature importance.
* Optimized Linear (Lasso), Random Forest, and XGBoost Regressors using **Optuna** to reach the best model.
* Built an **interactive API** application using Streamlit, which can be found [here](https://strava-kudos.herokuapp.com/).

## Motivation
Strava is a service for tracking human exercise which incorporates social network type features. It is mostly used for cycling and running, with an emphasis on using GPS data. I use Strava a lot as both a social media and to track my training. I've always been curious as to whether Kudos received on activities is inherently random, or if different attributes of that activity (e.g. run pace, or distance) affect Kudos in different ways. If I could build a Kudos prediction model that performed well on unseen data I would know that it isn't random! Also, this model would provide key insights into what different attributes affect kudos the most. 

Another motivation behind this project is that there are clear applications of this kind of modeling to a business use cases: for example, investigating how to maximize user interaction on a product.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, imblearn, shap, optuna, requests, pickle, seaborn, matplotlib<br />
**Requirements:** ```pip install -r requirements.txt```  
**Strava API:** https://towardsdatascience.com/using-the-strava-api-and-pandas-to-explore-your-activity-data-d94901d9bfde
**SHAP Article:** https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27  
**Optuna Hyperparameter Tuning:** https://medium.com/subex-ai-labs/efficient-hyperparameter-optimization-for-xgboost-model-using-optuna-3ee9a02566b1

## Project Write-Up
A blog post was written about this project and it was featured on Towards Data Science's editors pick section, it can be found [here](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053).

## Data Collection
Using Python and the Strava API I was able to automate the data collection stage. For each activity, some of the key attributes extracted are: kudos_count, name, distance, moving_time, total_elevation_gain, workout_type, start_date, location_city, photo_count, ...

## EDA
Some notable findings include:
* The Kudos received depended heavily on how many followers I had at the time. Unfortunately, there was no way to see how many followers I had at each point in time, therefore I could only use my most recent 1125 activities to train my model as the kudos stayed fairly consistent in this interval.
* It was found that the target variable is skewed right and there are some extreme values above ~100.
* Features such as distance, moving_time, and average_speed_mpk seem to share a similar distribution to the one we have with kudos_count.
* By looking at time distribution between activities, it was found that runs that are quickly followed in succession by other runs tend to receive fewer kudos than runs that were the only activity that day. To add to this, the longest activity of the day tends to receive more kudos than the other runs that day.

<p float="left">
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/pivot_table.png" width="200" />
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/indexs.png" width="300" /> 
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/corr.png" width="300" />
</p>

## Preprocessing and Feature Engineering
After obtaining the data, I needed to clean it up so that it was usable for my model. I made the following changes and created the following variables:
* An 80/20 train/test split was used and as my data contained dates, the most recent 20% of the data became the test set. I then split the training set into 5 folds using Sklearn's StratifiedKFold, Sturge's rule was used to bin the continuous target variable.
* Any missing values in a categorical feature were assigned a new category 'NONE' and missing values in numeric features were imputed using the median. Some heuristic functions were also used to impute systematic missing values. 
* Time-based features were added: year, month, day of the week, etc. Other features were also created using specific domain knowledge. I go into depth about this in the corresponding [blog post](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053).
* One-hot-encoding was used to encode categorical features and ordinal encoding was used to encode ordinal features. Target encoding was also used for a few categorical features. 

## Model Building 
In this step, I built a few different candidate models and compared different metrics to determine which was the best model for deployment. Three of those models were:
* Dummy Classifier (simply returns average kudos) - Baseline for the model.
* Lasso Regression - Because of the sparse data from the many categorical variables, I thought a normalized regression like lasso would be effective.
* Random Forest - Again, with the sparsity associated with the data, I thought that this would be a good fit.
* XGBRegressor - Well... this model just always seems to work.

Feature selection was performed using a mix of SHAP values and feature importance from XGB. 

Optuna was used to tune all three shortlisted models. In particular, the Tree-structured Parzen Estimator (TPE) was used.
<p align="center">
<img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/optuna.png" width="600" height="300">
</p>

## Model Performance
The XGB model far outperformed the other approaches on the test and validation sets.
* XGB Regressor : RMSE = 9.41
* Lasso Regression: RMSE = 10.54
* Random Forest Regressor: RMSE = 10.89

The final XGB model was then trained on the whole training dataset using the hyperparameters found in the Optuna experiment.

## Model Explainability
SHAP was used to interpret the model and individual predictions. It was found that longer and faster runs recieve more kudos (as they are more impressive?). Workouts also recieve more kudos than easy runs.

<p float="left">
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/shap_feature_imp.png" width="400" />
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/shap1.png" width="400" /> 
</p>

## Productionization
In this step, I built a [Streamlit app](https://strava-kudos.herokuapp.com/) that is hosted publicly using Heroku. The app uses the Strava API to get my most recent activities and makes real-time predictions on them.

<p align="center">
<img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/strava-streamlit.png" width="600" height="310">
</p>
