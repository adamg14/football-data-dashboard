import joblib
import pandas as pd
from read_data import premier_league_teams

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


def prediction_model(input_data):
    # Store original names for the final messages
    home_team_str = input_data["HomeTeam"][0]
    away_team_str = input_data["AwayTeam"][0]

    # Convert names to numeric codes
    home_team_index = premier_league_teams[home_team_str]
    away_team_index = premier_league_teams[away_team_str]
    input_data["HomeTeam"][0] = home_team_index
    input_data["AwayTeam"][0] = away_team_index

    input_data_frame = pd.DataFrame(input_data)

    model = joblib.load("./models/trained_classification_model.joblib")

    try:
        prediction = model.predict(input_data_frame)
        prediction_probability = model.predict_proba(input_data_frame)
        confidence = max(prediction_probability[0])

        if prediction == 1:
            return f"Prediction: {home_team_str}. Prediction confidence: {confidence:.2f}"
        elif prediction == -1:
            return f"Prediction: {away_team_str}. Prediction confidence: {confidence:.2f}"
        else:
            return f"Prediction: Draw. Prediction confidence: {confidence:.2f}"
    
    except Exception as e:
        # IMPORTANT: make it an f-string so we can see the real error
        return f"Error during prediction: {e}"