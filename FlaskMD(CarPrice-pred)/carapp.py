from flask import Flask, request, render_template
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load your trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Example mapping dictionaries for textual inputs
country_mapping = {
    'USA': 1, 'DE': 2, 'CN': 3, 'JP': 4, 'IN': 5, 'FR': 6, 'IT': 7, 'KR': 8,
    'CA': 9, 'GB': 10, 'RU': 11, 'BR': 12, 'AU': 13, 'MX': 14, 'ZA': 15, 'SA': 16
}
drive_mapping = {'FWD': 1, '4WD': 2, 'RWD': 3}

@app.route('/')
def home():
    return render_template('Car.html')

@app.route('/predict', methods=['POST'])
def predict():
    car_drive = request.form['car_drive']
    car_mileage = float(request.form['car_mileage'])
    car_country = request.form['car_country']
    car_engine_hp = float(request.form['car_engine_hp'])
    car_age = float(request.form['car_age'])

    # Convert textual input to numerical values using the mapping dictionaries
    car_drive_numeric = drive_mapping.get(car_drive, 0)  # Default to 0 if drive is not found
    car_country_numeric = country_mapping.get(car_country, 0)  # Default to 0 if country is not found

    
    # Creating user input array in logarithmic values
    user_input = np.array([[car_drive_numeric, car_mileage, car_country_numeric, car_engine_hp, car_age]])
    predicted_price = model.predict(user_input)

  
   

    return render_template('Car.html', prediction_text=f"The actual car price is: ₹{predicted_price[0]}")

if __name__ == '__main__':
    app.run(debug=True)
