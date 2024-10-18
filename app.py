from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

# Load the trained models
model_dict = joblib.load('all_models.pkl')

def fetch_historical_data(district, variety):
    # Load the historical data
    historical_data = pd.read_csv('agriculture_data.csv')

    # Convert 'Price Date' to datetime
    historical_data['Price Date'] = pd.to_datetime(historical_data['Price Date'])
    
    # Aggregate the mean price for each date, district, and variety
    historical_data = (historical_data
                       .groupby(['Price Date', 'District Name', 'Variety'])
                       .agg({'Capped Price': 'mean'})
                       .reset_index())

    # Set 'Price Date' as the index of the DataFrame
    historical_data.set_index('Price Date', inplace=True)

    # Filter the historical data for the specified district and variety
    historical_data = historical_data[(historical_data['District Name'] == district) & 
                                      (historical_data['Variety'] == variety)]
    
    return historical_data

def prepare_input_for_prediction(historical_data):
    # Check if the historical data DataFrame is empty
    if historical_data.empty:
        raise ValueError("Historical data is empty. Please check the data source.")

    # Resample and prepare historical data
    historical_data = historical_data['Capped Price'].resample('M').mean().ffill()

    # Create a DataFrame for processing
    df = pd.DataFrame({'Value': historical_data})

    # Create lag features
    for lag in range(1, 13):  # Lag features for the past 12 months
        df[f'lag_{lag}'] = df['Value'].shift(lag)

    # Add rolling features
    df['rolling_mean'] = df['Value'].rolling(window=3).mean()  # 3-month rolling mean
    df['rolling_std'] = df['Value'].rolling(window=3).std()    # 3-month rolling standard deviation

    # Drop rows with NaN values created by lags and rolling calculations
    df.dropna(inplace=True)

    if df.empty:
        raise ValueError("Not enough data to create lag features and rolling statistics. Please check the historical data.")

    # Apply Min-Max scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[['Value']])

    # Replace the original values with scaled values
    df['Value'] = scaled_data.flatten()

    return df, scaler

def predict_price(district, variety, future_date):
    # Get historical data for the district and variety
    historical_data = fetch_historical_data(district, variety)

    # Prepare the input data based on the historical data
    try:
        df, scaler = prepare_input_for_prediction(historical_data)
    except ValueError as e:
        print(e)
        return None

    # Retrieve the model for the specified district and variety
    model_info = model_dict.get((district, variety))

    if model_info is None:
        print("No model found for the specified district and variety.")
        return None

    model = model_info['model']

    # Use the most recent row of df (latest data) as input for the model
    if df.empty:
        print("No valid input data for prediction.")
        return None

    X_new = df.iloc[-1:].drop(columns=['Value'])

    # Make the prediction using the loaded model
    predicted_value = model.predict(X_new)

    # Inverse transform to convert the scaled value back to the original price scale
    predicted_price = scaler.inverse_transform(predicted_value.reshape(-1, 1))

    return predicted_price.flatten()[0]  # Flatten to get the scalar value

@app.route('/', methods=['GET', 'POST'])
def index():
    historical_data = []
    price = None

    if request.method == 'POST':
        district = request.form['district']
        variety = request.form['variety']
        future_date = request.form['future_date']
        
        # Fetch and prepare historical data
        try:
            historical_data_df = fetch_historical_data(district, variety)
            historical_data = historical_data_df.reset_index().rename(columns={'Price Date': 'date', 'Capped Price': 'value'}).to_dict(orient='records')
            
            # Predict price based on the provided inputs
            price = predict_price(district, variety, future_date)
        except Exception as e:
            print(f"Error fetching data or predicting price: {e}")
            historical_data = []
            price = None

    return render_template('index.html', price=price, historical_data=json.dumps(historical_data))

if __name__ == '__main__':
    app.run(debug=True)
