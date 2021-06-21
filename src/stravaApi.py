import pandas as pd
import requests

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

from pandas import json_normalize

auth_url = "https://www.strava.com/oauth/token"
activites_url = "https://www.strava.com/api/v3/athlete/activities"

payload = {
    'client_id': "50446",
    'client_secret': '7955204b7285824280b5b30af6361996b414dc9f',
    'refresh_token': '3d26f702b60d42195b267cd4029ed958143cb250',
    'grant_type': "refresh_token",
    'f': 'json'
}

# check if there is a new request token
print("Requesting Token...")
res = requests.post(auth_url, data=payload, verify=False)
access_token = res.json()['access_token']
print("\tAccess Token = {}\n".format(access_token))


header = {'Authorization': 'Bearer ' + access_token}

# initialise dataframe with first 200 activities
page_num = 1
param = {'per_page': 200, 'page': page_num}
my_dataset = requests.get(activites_url, headers=header, params=param).json()
activities = json_normalize(my_dataset)

# loop over pages until request fails
while True:
    page_num += 1

    if page_num % 10 == 0:
        print(f"Now extracting page number {page_num}")

    param = {'per_page': 200, 'page': page_num}
    my_dataset = requests.get(activites_url, headers=header, params=param).json()
    if my_dataset:
        new_activities = json_normalize(my_dataset)
        activities.append(new_activities)
    else:
        break

# save activities df to input folder
activities.to_csv('./input/strava_activities.csv')




