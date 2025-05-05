from flask import Flask, jsonify, request, render_template
import pymysql
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__)

# Configure logging
handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

# Database configuration with fallback values
db_config = {
    'host': os.getenv('DB_HOST', '153.92.13.207'),  # Fallback to direct value if env var not set
    'user': os.getenv('DB_USER', 'u346812618_kel2'),
    'password': os.getenv('DB_PASS', 'M0nitorcuaca65!'),
    'database': os.getenv('DB_NAME', 'u346812618_monitor_cuaca'),
    'cursorclass': pymysql.cursors.DictCursor,
    'connect_timeout': 10  # Add connection timeout
}

# Load the trained model with enhanced error handling
def load_weather_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'weather_model.pkl')
        if not os.path.exists(model_path):
            app.logger.error(f"Model file not found at: {model_path}")
            return None, None, None
            
        model_data = joblib.load(model_path)
        app.logger.info("Model loaded successfully")
        return model_data['model'], model_data['scaler'], model_data['features']
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}", exc_info=True)
        return None, None, None

model, scaler, features = load_weather_model()

def get_db_connection():
    try:
        connection = pymysql.connect(**db_config)
        app.logger.info("Database connection established")
        return connection
    except Exception as e:
        app.logger.error(f"Database connection failed: {str(e)}", exc_info=True)
        return None

def fetch_latest_sensor_data(hours=24):
    """Fetch latest sensor data with time range"""
    connection = get_db_connection()
    if not connection:
        return None
    
    try:
        query = """
        SELECT temperature, humidity, pressure as air_pressure, 
               lux as light_intensity, wind_speed, timestamp 
        FROM sensor_data 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        with connection.cursor() as cursor:
            cursor.execute(query, (hours,))
            result = cursor.fetchall()
            if not result:
                app.logger.warning("No sensor data found in database")
                return None
                
            df = pd.DataFrame(result)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            app.logger.info(f"Fetched {len(df)} records of sensor data")
            return df
    except Exception as e:
        app.logger.error(f"Error fetching sensor data: {str(e)}", exc_info=True)
        return None
    finally:
        connection.close()

def preprocess_for_prediction(last_day_data, features, scaler):
    """Prepare the latest data for prediction"""
    try:
        df = pd.DataFrame([last_day_data])
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year']/365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year']/365)
        
        # Calculate rolling means (using last 3 values if available)
        df['temp_rolling_mean'] = df['temperature']
        df['humidity_rolling_mean'] = df['humidity']
        
        # Feature engineering
        df['temp_humidity'] = df['temperature'] * df['humidity']
        df['pressure_change'] = 0  # Placeholder for actual calculation
        
        X = df[features]
        X_scaled = scaler.transform(X)
        return X_scaled
    except Exception as e:
        app.logger.error(f"Preprocessing error: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    try:
        prediction = get_weather_prediction()
        if not prediction:
            app.logger.warning("No prediction available for index page")
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('error.html'), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """New endpoint for debugging"""
    try:
        # Test database connection
        db_test = "Success" if get_db_connection() else "Failed"
        
        # Test model loading
        model_test = "Loaded" if model and scaler else "Failed"
        
        # Get sample data
        sample_data = fetch_latest_sensor_data(1)
        
        return jsonify({
            'status': 'debug',
            'database': db_test,
            'model': model_test,
            'sample_data': sample_data.iloc[0].to_dict() if sample_data is not None else None,
            'environment': {
                'DB_HOST': bool(os.getenv('DB_HOST')),
                'DB_USER': bool(os.getenv('DB_USER')),
                'DB_PASS': bool(os.getenv('DB_PASS')),
                'DB_NAME': bool(os.getenv('DB_NAME'))
            }
        })
    except Exception as e:
        app.logger.error(f"Debug error: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
