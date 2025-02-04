import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

gender_mapping = {'male': 0, 'female': 1}
class_mapping = {'economy': 1, 'business': 0, 'first': 2}
seat_mapping = {'aisle': 0, 'middle': 1, 'window': 2}

@app.route('/')
def home():
    return render_template('passenger.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def pred():
    Age = int(request.form.get('Age'))
    Gender = request.form.get('Gender')
    Class = request.form.get('Class')
    Seat_Type = request.form.get('Seat_Type')
    Fare_Paid = float(request.form.get('Fare_Paid'))

    Gen = gender_mapping.get(Gender, 0)
    classN = class_mapping.get(Class, 0)
    seatN = seat_mapping.get(Seat_Type, 0)

    feature = np.array([[Age, Gen, classN, seatN, Fare_Paid]])

    prediction = model.predict(feature)
    if prediction == 1:
        result = 'Survived'
    else:
        result = 'did not survive'

    return render_template('passenger.html', prediction_text=f'The passenger {result}')

if __name__ == '__main__':
    app.run(debug=True)
