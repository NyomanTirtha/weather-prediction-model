import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from datetime import timedelta
import joblib

def preprocess_data(df):
    """Preprocess data for machine learning"""
    df = df.copy()
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    df['temp_rolling_mean'] = df['temperature'].rolling(window=3, min_periods=1).mean()
    df['humidity_rolling_mean'] = df['humidity'].rolling(window=3, min_periods=1).mean()
    
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['pressure_change'] = df['air_pressure'].diff().fillna(0)
    
    features = ['temperature', 'humidity', 'air_pressure', 'light_intensity', 'wind_speed',
                'day_sin', 'day_cos', 'temp_rolling_mean', 'humidity_rolling_mean',
                'temp_humidity', 'pressure_change']
    
    X = df[features]
    y = df['weather_next_day']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler, features

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, 
                                 target_names=['Sunny', 'Cloudy', 'Partly Cloudy', 'Rainy', 'Overcast'])
    return acc, report

def predict_next_day(model, scaler, features, last_day_data):
    """Predict weather for next day"""
    df = pd.DataFrame([last_day_data])
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
    
    df['temp_rolling_mean'] = df['temperature']
    df['humidity_rolling_mean'] = df['humidity']
    
    df['temp_humidity'] = df['temperature'] * df['humidity']
    df['pressure_change'] = 0
    
    X = df[features]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    weather_map = {0: 'Sunny', 1: 'Cloudy', 2: 'Partly Cloudy', 3: 'Rainy', 4: 'Overcast'}
    return weather_map[pred[0]]

def save_model(model, scaler, features, filename='weather_model.pkl'):
    """Save model and preprocessing artifacts"""
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features
    }
    joblib.dump(model_data, filename)
    print(f"Model saved to {filename}")

def load_model(filename='weather_model.pkl'):
    """Load saved model"""
    model_data = joblib.load(filename)
    return model_data['model'], model_data['scaler'], model_data['features']

def main():
    # Load data
    data = pd.read_csv('weather_data.csv')
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    
    # Preprocess and train
    X, y, scaler, features = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_model(X_train, y_train)
    
    # Evaluate
    acc, report = evaluate_model(model, X_test, y_test)
    print(f"Model Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)
    
    # Save model
    save_model(model, scaler, features)
    
    # Example prediction
    last_day = data.iloc[-1]
    next_day_timestamp = last_day['timestamp'] + timedelta(days=1)
    last_day_data = {
        'temperature': last_day['temperature'],
        'humidity': last_day['humidity'],
        'air_pressure': last_day['air_pressure'],
        'light_intensity': last_day['light_intensity'],
        'wind_speed': last_day['wind_speed'],
        'timestamp': next_day_timestamp
    }
    prediction = predict_next_day(model, scaler, features, last_day_data)
    print(f"Predicted weather for next day ({next_day_timestamp.date()}): {prediction}")

if __name__ == "__main__":
    main()