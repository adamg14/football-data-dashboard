# match data from the 2020-21 Premier League Season
# data exploration / data preprocesing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

match_data = pd.read_csv("./soccer21-22.csv")
print(match_data.head())

# statistcal properties of the data frame
print(match_data.describe())

# checking for missing data - if there this data must be added manually by fetching 
# in this dataset there is no missing values
print(match_data.isnull().sum())

# refactoring the dataframe
# encoding the non-numeric columns to numeric columns
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

print(match_data.head())


# basic data visualisation - full-time home goals VS full-time away goals DISTRIBUTION
sns.histplot(match_data["FTHG"], kde=True, color='blue', label='Home Team Goals')
sns.histplot(match_data["FTAG"], kde=True, color='red', label='Away Team Goals')
plt.xlabel("Full-time Goals")
plt.ylabel("Count")
plt.legend()
plt.title("Distribution of Full-time Goals Classified by Home and Away Team")
plt.show()
plt.close()

# CORRELATION MATRIX
numeric_data_frame = match_data.drop(["Date", "Referee"], axis=1)
correlation_matrix = numeric_data_frame.corr()
cm_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, mask=cm_mask, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.xticks(rotation=90)
plt.show()
plt.close()