import pandas as pd
import shiny
from shiny import App, ui, render, reactive
import matplotlib.pyplot as plt
from data_exploration import match_data, numeric_data_frame
import seaborn as sns
from read_data import original_data
import numpy as np

app_ui = ui.page_fluid(
    ui.h1("Premier League 2021/2022 Match Data", style="text-align: center;"),
    ui.output_ui("rows", style="text-align: center;"),
    
    # first row of visualisations
    ui.row(
        ui.column(
            6,
            ui.h2("Match Data"),
            ui.output_ui("table")
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
    )
)



def server(input, output, session):
    @render.plot
    def histogram():
        fig, ax = plt.subplots()
        sns.histplot(match_data["FTHG"], kde=True, color='blue', label='Home Team Goals')
        sns.histplot(match_data["FTAG"], kde=True, color='red', label='Away Team Goals')
        plt.xlabel("Full-time Goals")
        plt.ylabel("Count")
        plt.legend()
        plt.title("Distribution of Full-time Goals Classified by Home and Away Team")
        return fig
    
    
    @render.table
    def table():
        return original_data.head()
    
    @render.plot
    def correlation_matrix_plot():
        numeric_data_frame = match_data.drop(["Date", "Referee"], axis=1)
        correlation_matrix = numeric_data_frame.corr()
        cm_mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        plt.figure(figsize=(10, 10))
        sns.heatmap(correlation_matrix, mask=cm_mask, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.xticks(rotation=90)
        
app = App(app_ui, server)