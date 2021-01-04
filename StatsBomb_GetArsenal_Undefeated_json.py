import sys
import urllib.request 
import json
import requests
import numpy as np
import pandas as pd

# https://stackoverflow.com/questions/15789059/python-json-only-get-keys-in-first-level
# https://stackoverflow.com/questions/12965203/how-to-get-json-from-webpage-into-python-script
# https://docs.python.org/3/library/json.html


matches_list = []

def ExploreJsonFile(data):

    # Use the json module to load the string data into a dictionary
    theJSON = json.loads(data)

    # List of Values that are dictionaries
    values_that_are_dictionaries = []
    
    #List of values that are strings
    values_that_are_strings = []

    dict_num = 0
    for dictionary in theJSON:
        for key, value in dictionary.items():
            if isinstance(value, dict):
                values_that_are_dictionaries.append(f" Key ------> {key} \n Values --> {value.keys()}")
                #print(f" Key ------> {key} \n Values --> {value.keys()}")
            else:
                values_that_are_strings.append(f"{key} with value: {value} not a dictionary !")
                #print(f"{key} with value: {value} not a dictionary !")

        List_All_Vals = create_lists_from_json(dictionary)
    
        dict_num += 1

    return List_All_Vals #, dict_num

matchdate_list = []
kick_off_list = []
competition_name_list = []
competition_country_list = []
comp_stage_list = []
season_year_list = []
home_team_list = [] 
home_team_scores_list = []  
away_team_list = []
away_team_scores_list = []


def create_lists_from_json(input_dictionary):

    matchdate_list.append(input_dictionary['match_date'])
    kick_off_list.append(input_dictionary['kick_off'])
    competition_name_list.append(input_dictionary['competition']['competition_name'])
    competition_country_list.append(input_dictionary['competition']['country_name'])
    comp_stage_list.append(input_dictionary['competition_stage']['name'])
    season_year_list.append(input_dictionary['season']['season_name'])
    home_team_list.append(input_dictionary['home_team']['home_team_name'])
    home_team_scores_list.append(input_dictionary['home_score'])  
    away_team_list.append(input_dictionary['away_team']['away_team_name'])
    away_team_scores_list.append(input_dictionary['away_score'])
    
    List_of_Lists = [matchdate_list, kick_off_list, competition_name_list, competition_country_list, comp_stage_list, season_year_list,
                home_team_list, home_team_scores_list, away_team_list, away_team_scores_list ]

    return List_of_Lists


List_of_Keys = ['match_date', 'kick_off', 'competition_name', 'country_name', 'competition_name', 'season_year', 'home_team_name',
                'home_score','away_team_name', 'away_score']


def create_dataframe(list_of_keys, list_of_lists): #, max_index):
    
    Resultant_Dictionary = {}
    
    for Dictionary_key, Dictionary_Value in zip(list_of_keys, list_of_lists):
        Resultant_Dictionary.update({Dictionary_key: Dictionary_Value})
        
    #DataFrame = pd.DataFrame(index=range(0, max_index), data=Resultant_Dictionary)
    DataFrame = pd.DataFrame(data=Resultant_Dictionary)

    #print(DataFrame)

    return DataFrame

matches = [2] 

seasons = [44] 

def main():
    # define a variable to hold the source URL
    for folder in matches:
        for link in seasons:
            urlData = f"https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/{folder}/{link}.json"
            try:
                webUrl = urllib.request.urlopen(urlData)
                matches_list.append(link)
                data = webUrl.read().decode("utf-8")
                all_desired_values = ExploreJsonFile(data)
                Dataframe = create_dataframe(List_of_Keys, all_desired_values) 

            except urllib.error.HTTPError:
                continue

    return Dataframe


if __name__ == "__main__":
    main()
    



