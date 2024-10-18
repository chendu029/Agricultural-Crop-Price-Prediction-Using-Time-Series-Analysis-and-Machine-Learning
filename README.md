# chendu029-Agricultural-Crop-Price-Prediction-Using-Time-Series-Analysis-and-Machine-Learning

Here's a sample README.md file for your project, which you can use for your GitHub repository:

markdown
Copy code
# Agricultural Crop Price Prediction

## Overview

This project aims to predict the future prices of agricultural crops using historical price data and machine learning techniques. It leverages time series analysis to model price trends and provides an interactive web interface for users to input their queries.

## Functionalities

- **Data Retrieval**: Fetches historical crop price data from a SQLite database based on user-selected district and crop variety.
- **Data Processing**: Processes the historical data by:
  - Resampling and filling missing values.
  - Creating lag features for the past 12 months.
  - Adding rolling statistics (mean and standard deviation) to enhance the model input.
- **Machine Learning Model**: Utilizes trained machine learning models (loaded from a pickle file) to predict future prices.
- **Interactive Web Interface**: A Flask-based web application that allows users to:
  - Input the district, crop variety, and future date for price prediction.
  - View historical price data in a visually appealing format (line plots).
  - Receive predicted future prices based on their input.

## Technologies Used

- **Python**: The primary programming language for the application.
- **Flask**: A lightweight WSGI web application framework for building the web interface.
- **Pandas**: Used for data manipulation and analysis.
- **NumPy**: Provides support for large, multi-dimensional arrays and matrices.
- **Scikit-learn**: A library for machine learning that provides tools for model training and evaluation.
- **SQLite**: A lightweight database engine for storing historical price data.
- **Joblib**: For loading and saving trained machine learning models.
- **Matplotlib**: For plotting historical price data and predictions.
