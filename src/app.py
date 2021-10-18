import requests
import pickle
from requests.sessions import default_headers
import urllib3
import re
import polyline
import shap

import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

st.set_option("deprecation.showPyplotGlobalUse", False)


import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_squared_error, r2_score
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import plotly.express as px

import matplotlib.pyplot as plt
import seaborn as sns

# from plotly.subplots import make_subplots
import config
from preprocess import preprocess_unseen
from predict import predict, process


def uk_awake(hour):
    if hour >= 6 and hour <= 19:
        return 1
    else:
        return 0


@st.cache(allow_output_mutation=True)
def load_strava_data():
    train_data = pd.read_csv(config.STRAVA_TRAIN_PATH)
    train_data.loc[:, "datetime"] = pd.to_datetime(
        train_data["GMT_date"] + " " + train_data["GMT_time"]
    )
    train_data.loc[:, "hour"] = train_data["datetime"].dt.hour
    train_data.loc[:, "uk_awake"] = train_data.hour.apply(uk_awake)
    y = train_data.kudos_count
    train_data = process(train_data)
    return train_data, y


@st.cache
def load_recent_data(recent=10):
    # get payload information
    auth_url = "https://www.strava.com/oauth/token"
    activites_url = "https://www.strava.com/api/v3/athlete/activities"
    payload = config.STRAVA_API

    # check if there is a new request token
    res = requests.post(auth_url, data=payload, verify=False)
    access_token = res.json()["access_token"]
    header = {"Authorization": "Bearer " + access_token}

    # initialise dataframe with first 5 activities
    param = {"per_page": recent * 2, "page": 1}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()
    activities = pd.json_normalize(my_dataset)
    # keep useful features
    columns_to_keep = [
        "name",
        "kudos_count",
        "distance",
        "moving_time",
        "elapsed_time",
        "total_elevation_gain",
        "type",
        "workout_type",
        "start_date",
        "start_date_local",
        "timezone",
        "start_latlng",
        "end_latlng",
        "achievement_count",
        "comment_count",
        "athlete_count",
        "photo_count",
        "manual",
        "private",
        "average_speed",
        "max_speed",
        "average_heartrate",
        "max_heartrate",
        "pr_count",
        "total_photo_count",
        "suffer_score",
        "map.summary_polyline",
    ]
    activities = activities[columns_to_keep].head(recent)
    return activities


def load_model():
    with open("models/production/xgb_model.pickle", "rb") as f:
        model = pickle.load(f)
    return model


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def app_main():

    st.title("Kudos Predictor :running: :dart:")
    st.subheader(
        "Predicting user interacton on my own [Strava](https://www.strava.com/athletes/5028644) activities based on the respective activity attributes."
    )

    # load in xgb model
    model = load_model()

    st.write("---")

    st.write(
        ":arrow_down_small: Predict Kudos on most recent activities :arrow_down_small:"
    )
    # slider to select number of datapoints
    datapoints = st.slider(
        "How many recent activites to predict on?", min_value=2, max_value=50, value=50
    )
    predict_button = st.button(f"Predict on {datapoints} new data points")
    if predict_button:
        with st.spinner("Wait for it..."):
            new_data = load_recent_data(recent=datapoints * 2)
            new_data_preprocessed = preprocess_unseen(new_data).head(datapoints)
            # st.write("Unprocessed raw data :arrow_down:")
            # st.write(new_data.head(datapoints))
            st.write("")
            st.write("")
            st.write(":arrow_down_small: Data with predictions :arrow_down_small:")
            new_preds = predict(new_data_preprocessed, model=model)
            new_data_preds = new_data_preprocessed
            new_data_preds.loc[:, "Predicted Kudos"] = new_preds

            new_data_preds.loc[:, "Error"] = abs(
                new_data_preds["kudos_count"] - new_data_preds["Predicted Kudos"]
            )
            cols = [
                "local_date",
                "name",
                "kudos_count",
                "Predicted Kudos",
                "Error",
                "distance",
                "moving_time",
                "total_elevation_gain",
                "suffer_score",
                "average_heartrate",
                "max_heartrate",
                "run_area",
                "pr_count",
                "workout_type",
                "is_named",
                "run_per_day",
                "max_run",
                "uk_awake",
            ]
            rename = {
                "local_date": "Date",
                "name": "name",
                "kudos_count": "Kudos",
                "Predicted Kudos": "Predicted Kudos",
                "Error": "Error",
                "distance": "Distance (km)",
                "moving_time": "Moving Time (min)",
                "total_elevation_gain": "Total climb (m)",
                "suffer_score": "Suffer Score",
                "average_heartrate": "Average Heartrate",
                "max_heartrate": "Max Heartrate",
                "run_area": "Area Enclosed By Run",
                "pr_count": "PR Count",
                "workout_type": "Workout Type",
                "is_named": "Is Run Named?",
                "run_per_day": "Number of Runs That Day",
                "max_run": "Longest Run of Day?",
                "uk_awake": "Is The U.K. Awake?",
            }
            new_data_preds = new_data_preds[cols]

            display_preds = new_data_preds.rename(rename, axis="columns")
            display_preds = display_preds.rename(columns={"Date": "index"}).set_index(
                "index"
            )
            st.write(display_preds)
            rmse = mean_squared_error(
                new_data_preds.kudos_count[1:], new_preds[1:], squared=False,
            )
            st.write(
                f"This unseen dataset consisting of {datapoints} data points has a mean Kudos value of {round(np.mean(new_data_preds.kudos_count),2)} with a standard deviation of {round(np.std(new_data_preds.kudos_count),2)}."
            )
            st.write("---")
            col1, col2 = st.columns(2)
            col1.metric(label="Root Mean Squared Error", value=f"{round(rmse, 2)}")
            col2.metric(
                label="R2 Score",
                value=f"{round(r2_score(new_data_preds.kudos_count[1:], new_preds[1:]),2)}",
            )
            st.write("---")
            if display_preds.shape[0] > 10:
                # Add histogram data
                hist_data = display_preds[["Kudos", "Predicted Kudos"]]
                st.write("True Kudos Values vs. Predicted")
                st.line_chart(data=hist_data, use_container_width=True)


def app_explain():
    st.title("Explaining The Model :question: :thought_balloon:")
    st.write(
        """The XGBOOST model was trained on a dataset consisting of 900 datapoints (test set had 225). 
    Shapely Additive Explanations (ShAP) were used for feature importance. A SHAP value is computed for every
    feature and these values are the average marginal contribution for the feature value across all the possible combinations of features."""
    )
    st.write(
        "The final features the model was trained on are as follows (target encodings were also added for the categorical features):"
    )
    col1, col2 = st.columns(2)
    col1.write(
        """ 
            - **Distance**: Distance of corresponding activity (km)
            - **Avergage Speed**: Average speed of activity (min/km)
            - **Max Speed**: Maximum speed of activity 
            - **Moving Time**: Moving time of activity (min)
            - **Max Heartrate**: Maximum heartrate of activity
            - **Total Elevation Gain**: Meteres climbed in the activity 
            - **Run Area**: Area enclosed by the activity (a bigger loop has a bigger run area)

        """
    )
    col2.write(
        """ 
            - **Max Run**: (Categorical) Is the activity the longest activity of the day?
            - **Workout Type**: (Categorical) Type of activity. 0 is an easy run, 1 is a race, 2 is a long run, and 3 is a workout. 
            - **Is Named**: (Categorical) Is the activity named on Strava?
            - **Run Per Day**: (Categorical) How many activities were done that day?
            - **UK Awake**: (Categorical) Is the UK awake when this acitivity was done?
        """
    )
    # load in xgb model
    model = load_model()
    # load data
    with st.spinner("Wait for it..."):
        new_data = load_recent_data(recent=50 * 2)
        new_data_preprocessed = preprocess_unseen(new_data).head(50)
        X = process(new_data_preprocessed)

    # shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    st.subheader("Explaining single prediction")
    st.write(
        """
    With SHAP, we can generate explanations for a single prediction. The SHAP plot shows features that contribute to pushing the output from the base value (average model output) to the actual predicted value.
    Red color indicates features that are pushing the prediction higher, and blue color indicates just the opposite.

    """
    )
    # pick datapoint
    point = st.slider("Pick a datapoint", 1, 50, 25)
    idx = point - 1
    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    st_shap(
        shap.force_plot(explainer.expected_value, shap_values[idx, :], X.iloc[idx, :])
    )

    # load in full training set
    train_data, _ = load_strava_data()
    shap_values_all = explainer.shap_values(train_data)

    st.subheader("Assessing feature importance based on Shap values")
    st.pyplot(
        shap.summary_plot(shap_values_all, train_data, plot_type="bar"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0,
    )

    st.subheader("Explaining the dataset")

    st.write(
        """
                We can visualize the importance of the features and their impact on the prediction by plotting summary charts. The one below sorts features by the sum of SHAP value magnitudes over all samples. It also uses SHAP values to show the distribution of the impacts each feature has.
                The color represents the feature value — red indicating high and blue indicating low. 

            """
    )

    st.pyplot(
        shap.summary_plot(shap_values_all, train_data),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0,
    )
    st.subheader("Explaining single feature")
    st.write(
        """
    Explaining single feature
    To understand the effect a single feature has on the model output, we can plot a SHAP value of that feature vs. the value of the feature for all instances in the dataset.
    The chart below shows the change in kudos count as the feature value changes. Vertical dispersions at a single value show interaction effects with other features. 
    SHAP automatically selects another feature for coloring to make these interactions easier to see:

    """
    )
    select_feature = st.selectbox(
        "What feature would you like to see?", train_data.columns
    )
    st.pyplot(
        shap.dependence_plot(select_feature, shap_values_all, train_data),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0,
    )


def app_EDA():
    st.title("Exploratory data analysis ::mag_right: :flashlight:")
    # load in full training set
    st.write("---")

    st.header("**Input DataFrame**")
    df, y = load_strava_data()
    df.loc[:, "kudos_count"] = y
    st.write(df)
    st.write("---")

    st.header("Correlation Matrix")
    sns.set_theme(style="white")
    # Compute the correlation matrix
    corr = df.corr()
    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    # Draw the heatmap with the mask and correct aspect ratio
    fig = plt.figure()
    matrix = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=0.3,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    st.pyplot(fig)

    pr = ProfileReport(df, explorative=True, correlations=None)

    st.write("---")
    st.header("**Pandas Profiling Report**")
    st_profile_report(pr)


def app():
    PAGES = {
        "Kudos Predicion": app_main,
        "Interpreting The Model": app_explain,
        "EDA": app_EDA,
    }
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()), index=0)
    PAGES[selection]()


if __name__ == "__main__":
    app()
