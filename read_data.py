# match data from the 2020-21 Premier League Season
# data exploration / data preprocesing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

match_data = pd.read_csv("./soccer21-22.csv")
original_data  = pd.read_csv("./soccer21-22.csv")
premier_league_teams = {
    "Arsenal": 0,
    "Aston Villa": 1,
    "Brentford": 2,
    "Burnley": 3,
    "Brighton": 4,
    "Chelsea": 5,
    "Crystal Palace": 6,
    "Everton": 7,
    "Leeds": 8,
    "Leicester": 9,
    "Liverpool": 10,
    "Man City": 11,
    "Man United": 12,
    "Newcastle": 13,
    "Norwich": 14,
    "Southampton": 15,
    "Tottenham": 16,
    "Watford": 17,
    "West Ham": 18,
    "Wolves": 19
}

result_map = {
    "D": 0,
    "H": -1,
    "A": 1
}

match_data["HomeTeam"] = match_data["HomeTeam"].map(premier_league_teams)
match_data["AwayTeam"] = match_data["AwayTeam"].map(premier_league_teams)
match_data["FTR"] = match_data["FTR"].map(result_map)
match_data["HTR"] = match_data["HTR"].map(result_map)

numeric_data_frame = match_data.drop(["Date", "Referee"], axis=1)