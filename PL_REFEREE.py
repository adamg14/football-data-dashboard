import pandas as pd

data_frame = pd.read_csv("./soccer21-22.csv")
PL_REFEREE = data_frame["Referee"].unique().tolist()