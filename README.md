# HeatingLoad-Prediction
A machine learning project that predicts building heating load using Polynomial Regression. Features an interactive Streamlit web app for local deployment, allowing users to input building parameters and visualize predicted heating energy requirements.

This project focuses on predicting the heating load of a building — the amount of energy required to maintain indoor comfort — using Polynomial Regression, a powerful technique for modeling nonlinear relationships.

Overview

Energy efficiency is one of the most important aspects of sustainable building design. The heating load depends on several architectural and environmental factors such as wall area, roof area, glazing area, and building orientation. This project aims to develop a regression-based model that accurately predicts the heating load using these input parameters.

Methodology

->Data Preprocessing: The dataset was cleaned and normalized to remove inconsistencies and scale numerical features.

->Model Selection: Polynomial Regression was chosen to capture the nonlinear relationship between features and the target variable (heating load).

->Model Training & Evaluation: The model was trained on a labeled dataset, and performance was evaluated using common regression metrics such as R² Score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

->Visualization: Key visualizations include scatter plots, polynomial fit curves, and residual analysis to assess model performance.

Deployment

To make the model interactive and user-friendly, a Streamlit web app was developed and deployed on localhost. The app allows users to input building parameters and instantly get the predicted heating load, making it easy to experiment with different configurations.

Key Features

->Interactive user interface built with Streamlit

->Polynomial Regression for nonlinear prediction

->Visual performance analysis of model fit

->Local deployment for testing and demonstration

Tech Stack

->Python

->Scikit-learn

->Pandas, NumPy, Matplotlib

->Streamlit
