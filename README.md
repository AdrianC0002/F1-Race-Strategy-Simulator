# Formula 1 Race Strategy Simulator

## Project Overview

This project uses machine learning to analyse Formula 1 race performance and compare race strategy options. The main aim is to predict lap times using race-related data and then use these predictions to simulate different tyre and pit-stop strategies.

The project focuses on simplified dry-race conditions. It does not try to fully recreate every real Formula 1 situation, such as safety cars, crashes, wet weather, traffic, or team tactics. Instead, it provides a structured and interpretable way to compare selected race strategies.

## Research Aim

The main research question for this project is:

**To what extent can machine learning-based lap-time prediction support realistic and effective Formula 1 race-strategy optimisation under simplified dry-race conditions?**

The project separates the problem into two stages:

1. Predict lap times using machine learning.
2. Use the predicted lap times to compare race strategies.

This makes the approach easier to understand than building one complex model that directly chooses a strategy.

## Dataset

The data was collected using the FastF1 Python package. The dataset contains lap-level Formula 1 race data from five selected tracks across the 2022 to 2025 seasons:

- Bahrain
- Saudi Arabia
- Spain
- Italy / Monza
- Abu Dhabi

Each row represents one driver’s lap in one race. Important features include:

- Grand Prix
- Driver
- Team
- Lap number
- Stint
- Tyre life
- Tyre compound
- Lap time

The final cleaned dataset was used to train and evaluate the machine learning models.

## Machine Learning Models

Three models were compared:

- Dummy baseline
- Ridge Regression
- XGBoost

The dummy model was used as a simple benchmark. Ridge Regression was used because it is easy to interpret. XGBoost was used because it can model more complex relationships in tabular data.

Model performance was evaluated using:

- MAE
- RMSE

Both Ridge Regression and XGBoost performed better than the dummy baseline. XGBoost was selected for the final simulator because it performed slightly better on the most recent unseen test season.

## Strategy Simulator

The trained XGBoost model was deployed in a Streamlit dashboard. The user can select a track, enter a tyre compound order, choose pit laps, and run a race strategy simulation.

The application outputs:

- Predicted total race time
- Lap-by-lap predicted lap times
- Predicted lap-time graph
- Cumulative race-time graph

The simulator currently uses a fixed driver-team case: **Lando Norris / McLaren**.

## Deployment

The application was deployed using Streamlit Cloud and can be accessed here:

https://f1-race-strategy-simulator-bxnkbqnerbyuksqygtornz.streamlit.app/

## Limitations

This project is a simplified decision-support tool. It does not include real-time race events, weather, safety cars, traffic, fuel load, or uncertainty estimates. Therefore, the results should not be treated as exact race strategy recommendations.

## Technologies Used

- Python
- pandas
- NumPy
- scikit-learn
- XGBoost
- FastF1
- Streamlit
- Matplotlib
- Joblib

## Conclusion

This project shows that machine learning can be used to support Formula 1 strategy analysis. The final system provides a clear and practical way to compare selected dry-race strategies using predicted lap times.
