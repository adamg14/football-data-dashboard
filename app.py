import pandas as pd
import shiny
from shiny import App, ui, render, reactive, req
import matplotlib.pyplot as plt
from data_exploration import match_data, numeric_data_frame
import seaborn as sns
from read_data import original_data
import numpy as np
from PL_TEAMS import PL_TEAMS
from PL_REFEREE import PL_REFEREE
from faicons import icon_svg
from prediction_model import prediction_model

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_select(
            id="team",
            label="Filter by team",
            choices=PL_TEAMS
        ),
        
    ui.h5("Machine Learning Model Match Prediction", style="text-align: center;"),
    ui.h6("Enter match statistics and the machine learning model will deterimine the result of the game"),
    
    ui.input_select(
        id="HomeTeam",
        label="Home Team",
        choices=PL_TEAMS,
        selected="Arsenal"
    ),
            
    ui.input_select(
        id="AwayTeam",
        label="Away Team",
        choices=PL_TEAMS,
        selected="Aston Villa"
    ),
            
    ui.input_numeric(
        id="HomeTeamHalfGoals",
        label="Home Team Half Time Goals",
        value=1,
        min=0,
        max=10,
        step=1
    ),
            
    ui.input_numeric(
        id="AwayTeamHalfGoals",
        label="Away Team Half Time Goals",
        value=1,
        min=0,
        max=10,
        step=1
    ),
            
    ui.input_numeric(
        id="HomeShots",
        label="Home Team Shots",
        value=3,
        min=0,
        max=50,
        step=1
    ),
            
    ui.input_numeric(
        id="AwayShots",
        label="Away Team Shots",
        value=3,
        min=0,
        max=50,
        step=1
    ),
            
    ui.input_numeric(
        id="HomeShotsTarget",
        label="Home Team Shots on Target",
        value=2,
        min=0,
        max=50,
        step=1
    ),
            
    ui.input_numeric(
        id="AwayShotsTarget",
        label="Away Team Shots on Target",
        value=2,
        min=0,
        max=50,
        step=1
    ),
            
    ui.input_numeric(
        id="HomeFouls",
        label="Home Team Fouls",
        value=2,
        min=0,
        max=40,
        step=1
    ),
            
    ui.input_numeric(
        id="AwayFouls",
        label="Away Team Fouls",
        value=2,
        min=0,
        max=40,
        step=1
    ),
            
    ui.input_numeric(
        id="HomeCorners",
        label="Home Team Corners",
        value= 3,
        min=0,
        max=20
    ),
            
    ui.input_numeric(
        id="AwayCorners",
        label="Home Team Corners",
        value= 1,
        min=0,
        max=20
    ),
            
    ui.input_numeric(
        id="HomeYellow",
        label="Home Team Yellow Cards",
        value=1,
        min=0,
        max=16
    ),
            
    ui.input_numeric(
        "AwayYellow",
        label="Away Team Yellow Cards",
        value=2,
        min=0,
        max=16
    ),
            
    ui.input_numeric(
        id="HomeRed",
        label="Home Team Red Cards",
        value=0,
        min=0,
        max=4
    ),
            
    ui.input_numeric(
        id="AwayRed",
        label="Away Team Red Cards",
        value=0,
        min=0,
        max=4
    ),
    
    ),
    
    ui.page_fluid(
    ui.h1("Premier League 2021/2022 Match Data", style="text-align: center;"),
    ui.output_ui("rows", style="text-align: center;"),
    
    # first row of visualisations
    ui.row(
        ui.navset_card_tab(
        ui.nav_panel(
            ui.output_data_frame("table"),
            icon=icon_svg("table"),
            
        ),
        ),
    
    # second row of visualisation
    ui.row(
        ui.column(
            6,
            ui.h2("Variable Correlation Matrix"),
            ui.output_plot("correlation_matrix_plot")
        ),
        ui.column(
            6,
            ui.h2("Goal Distribution Histogram: Home vs Away"),
            ui.output_plot("histogram")
        ),
    ),
    
    ui.row(
        ui.h2("Result"),
        ui.output_text("model_result")
    ),
)   
)
)



def server(input, output, session):
    @render.plot
    def histogram():
        if input.team() == "All Teams":
            fig, ax = plt.subplots()
            sns.histplot(match_data["FTHG"], kde=True, color='blue', label='Home Team Goals')
            sns.histplot(match_data["FTAG"], kde=True, color='red', label='Away Team Goals')
            plt.xlabel("Full-time Goals")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Distribution of Full-time Goals Classified by Home and Away Team")
        else:
            filtered_match_data = match_data[(original_data["HomeTeam"] == input.team()) | (original_data["AwayTeam"] == input.team())]
            fig, ax = plt.subplots()
            sns.histplot(filtered_match_data["FTHG"], kde=True, color='blue', label='Home Team Goals')
            sns.histplot(filtered_match_data["FTAG"], kde=True, color='red', label='Away Team Goals')
            plt.xlabel("Full-time Goals")
            plt.ylabel("Count")
            plt.legend()
            plt.title("Distribution of Full-time Goals Classified by Home and Away Team")
        return fig
    
    
    @render.data_frame
    def table():
        if input.team() == "All Teams":
            return render.DataGrid(original_data)
        else:
            team_filter = original_data[(original_data["HomeTeam"] == input.team()) | (original_data["AwayTeam"] == input.team())]
            return render.DataGrid(team_filter)
    
    
    @render.plot
    def correlation_matrix_plot():
        numeric_data_frame = match_data.drop(["Date", "Referee"], axis=1)
        
        if input.team() == "All Teams":
            correlation_matrix = numeric_data_frame.corr()
        else:
            correlation_matrix = numeric_data_frame[((original_data["HomeTeam"] == input.team()) | (original_data["AwayTeam"] == input.team()))].corr()
        cm_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation_matrix, mask=cm_mask, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.xticks(rotation=90)
    
    
    @render.text
    def model_result():
        input_data = {
            "HomeTeam": [input.HomeTeam()],
            "AwayTeam": [input.AwayTeam()],
            "HTHG": [input.HomeTeamHalfGoals()],
            "HTAG": [input.AwayTeamHalfGoals()],
            "HS": [input.HomeShots()],
            "AS": [input.AwayShots()],
            "HST": [input.HomeShotsTarget()],
            "AST": [input.AwayShotsTarget()],
            "HF": [input.HomeFouls()],
            "AF": [input.AwayFouls()],
            "HC": [input.HomeCorners()],
            "AC": [input.AwayCorners()],
            "HY": [input.HomeYellow()],
            "AY": [input.AwayYellow()],
            "HR": [input.HomeRed()],
            "AR": [input.AwayRed()],
        }
        
        return prediction_model(input_data)
    
app = App(app_ui, server)