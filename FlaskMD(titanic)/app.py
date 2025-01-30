from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)


with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('titanic.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = float(request.form['age'])
    fare = float(request.form['fare'])
    sibsp = int(request.form['sibsp'])
    parch = int(request.form['parch'])

   
    features = np.array([[age, fare, sibsp, parch]])
    prediction = model.predict(features)
    prediction_text = 'Survived' if prediction[0] == 1 else 'Did not survive'

    return render_template('titanic.html', Prediction_text=f'Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
