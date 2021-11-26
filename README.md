# Strava Kudos Predictor: Project Overview
## Project Overview
* Created a tool that **predicts kudos** (a proxy for user interaction) on [Strava activities](https://www.strava.com/athletes/5028644) (RMSE: 9.41) to see if it was random or if different attributes impact kudos in different ways.
* Attained over 4000 Strava activities using the **Strava API** and python.
* **Engineered new features** using domain knowledge. Features encapsulating different run types and times of day were added, for example.
* Performed **feature selection** using a combination of SHAP values and feature importance.
* Optimized Linear, Random Forest, and XGBoost Regressors using **Optuna** to reach the best model.
* Built an **interactive API** application using Streamlit.

## Motivation
Strava is a service for tracking human exercise which incorporates social network type features. It is mostly used for cycling and running, with an emphasis on using GPS data. I use Strava a lot as both a social media and to track my training. I've always been curious as to whether Kudos received on activities is inherently random, or if different attributes of that activity (e.g. run pace, or distance) affect Kudos in different ways. If I could build a Kudos prediction model that performed well on unseen data I would know that it isn't random! Also, this model would provide key insights into what different attributes affect kudos the most. 

Another motivation behind this project is that there are clear applications of this kind of modeling to a business use case: for example investigating how to maximize user interaction on a product.

## Code and Resources Used 
**Python Version:** 3.7  
**Packages:** pandas, numpy, sklearn, imblearn, shap, optuna, requests, pickle, seaborn, matplotlib 
**For Web Framework Requirements:**  ```pip install -r requirements.txt```  
**Strava API:** https://towardsdatascience.com/using-the-strava-api-and-pandas-to-explore-your-activity-data-d94901d9bfde
**SHAP Article:** https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27  
**Optuna Hyperparameter Tuning:** https://medium.com/subex-ai-labs/efficient-hyperparameter-optimization-for-xgboost-model-using-optuna-3ee9a02566b1

## Project Write-Up
A blog post was written about this project and it was featured on Towards Data Science's editors pick section, it can be found [here](https://towardsdatascience.com/predicting-strava-kudos-1a4ce7a02053).

## Data Collection
Using Python and the Strava API I was able to automate the data collection stage. For each activity, some of the key attributes extracted are: kudos_count, name, distance, moving_time, total_elevation_gain, workout_type, start_date, location_city, photo_count, ...

## EDA
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights. 

<p float="left">
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/pivot_table.png" width="200" />
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/indexs.png" width="300" /> 
  <img src="https://github.com/jackmleitch/StravaKudos/blob/main/input/images/corr.png" width="300" />
</p>

## Data Cleaning and Feaure Engineering
After gathering the data, I needed to clean it up so it was usable for my model. 

