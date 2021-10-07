import requests
import pickle
import urllib3
import re
import polyline

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# import plotly.express as px
# import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# from plotly.subplots import make_subplots
from preprocess import preprocess_unseen
from predict import predict


@st.cache
def load_recent_data(recent=10):
    # get payload information
    auth_url = "https://www.strava.com/oauth/token"
    activites_url = "https://www.strava.com/api/v3/athlete/activities"
    STRAVA_PAYLOAD = {
        "client_id": "50446",
        "client_secret": "7955204b7285824280b5b30af6361996b414dc9f",
        "refresh_token": "3d26f702b60d42195b267cd4029ed958143cb250",
        "grant_type": "refresh_token",
        "f": "json",
    }
    payload = STRAVA_PAYLOAD

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
    activities = activities[columns_to_keep].head(10)
    return activities


@st.cache
def load_model():
    with open("models/production/xgb_model.pickle", "rb") as f:
        model = pickle.load(f)
    return model


def app():

    st.title("Kudos Prediction :running: :dart:")
    st.subheader(
        "Predicting user interacton (Kudos given) on my own [Strava](https://www.strava.com/athletes/5028644) activities based on the respective activities attributes."
    )

    # load in xgb model
    model = load_model()

    # new data button
    get_new_data = st.button("Get most recent 5 activities")
    if get_new_data:
        with st.spinner("Wait for it..."):
            new_data = load_recent_data()
            new_data_preprocessed = preprocess_unseen(new_data).head(5)
        st.write("Unprocessed raw data :arrow_down:")
        st.write(new_data.head(5))
        st.write("Preprocessed data :arrow_down:")
        st.write(new_data_preprocessed)

    # predict kudos on new data
    # preds = predict(new_data_preprocessed, model=model)
    # print(preds)


if __name__ == "__main__":
    app()
