# app.py
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('titanic_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sex = 1 if request.form['sex'] == 'female' else 0
    fare = float(request.form['fare'])
    prediction = model.predict([[sex, fare]])[0]
    result = 'Survived' if prediction == 1 else 'Did not survive'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
