import pandas as pd
import requests

import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pandas import json_normalize

import config

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

payload = config.STRAVA_API

# check if there is a new request token
print("Requesting Token...")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()["access_token"]
print("\tAccess Token = {}\n".format(access_token))


header = {"Authorization": "Bearer " + access_token}

# initialise dataframe with first 200 activities
page_num = 1
param = {"per_page": 200, "page": page_num}
my_dataset = requests.get(activites_url, headers=header, params=param).json()
activities = json_normalize(my_dataset)

# print(activities.start_date[0])

# loop over pages until request fails
while True:
    page_num += 1

    if page_num % 5 == 0:
        print(f"Now extracting page number {page_num}")

    param = {"per_page": 200, "page": page_num}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()
    if my_dataset:
        new_activities = json_normalize(my_dataset)
        # activities.append(new_activities)
        activities = activities.append(new_activities, ignore_index=True)

    else:
        break


# save activities df to input folder
activities.to_csv("./input/strava_activities.csv")
