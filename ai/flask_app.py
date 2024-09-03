from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
# Load models
def load_model(model_path):
    return joblib.load(model_path)

# Flask app initialization
app = Flask(__name__)
CORS(app)
# Load models when the app starts
diabetes_model = load_model(os.path.join('../ai', 'diabetes_model.pkl'))
breast_cancer_model = load_model(os.path.join('../ai', 'breast_model.pkl'))
heart_disease_model = load_model(os.path.join('../ai', 'heart_model.pkl'))

# Define the predictive system function for diabetes
def predictive_system_diabetes(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_data = (Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    predictions = diabetes_model.predict(input_data_reshaped)
    if predictions[0] == 0:
        return "No Diabetes Detected"
    else:
        return "Diabetes Detected"

# Define the predictive system function for heart disease
def predictive_system_heart(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    predictions = heart_disease_model.predict(input_data_reshaped)
    if predictions[0] == 0:
        return "Healthy Heart"
    else:
        return "Have a Heart Disease"

# Define the predictive system function for breast cancer
def predictive_system_breast(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a1, b1, c1, d1):
    input_data = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a1, b1, c1, d1)
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    predictions = breast_cancer_model.predict(input_data_reshaped)
    if predictions[0] == 0:
        return "The Breast Cancer is Malignant"
    else:
        return "The Breast Cancer is Benign"

# Endpoint for Diabetes Prediction
@app.route('/predict/diabetes', methods=['POST'])
def predict_diabetes():
    data = request.json
    result = predictive_system_diabetes(
        Pregnancies=data['Pregnancies'],
        Glucose=data['Glucose'],
        BloodPressure=data['BloodPressure'],
        SkinThickness=data['SkinThickness'],
        Insulin=data['Insulin'],
        BMI=data['BMI'],
        DiabetesPedigreeFunction=data['DiabetesPedigreeFunction'],
        Age=data['Age']
    )
    return jsonify({'prediction': result})

# Endpoint for Breast Cancer Prediction
@app.route('/predict/breastcancer', methods=['POST'])
def predict_breast_cancer():
    data = request.json
    result = predictive_system_breast(
        a=data['a'], b=data['b'], c=data['c'], d=data['d'], e=data['e'], f=data['f'],
        g=data['g'], h=data['h'], i=data['i'], j=data['j'], k=data['k'],l=data['l'], m=data['m'],
        n=data['n'], o=data['o'], p=data['p'], q=data['q'], r=data['r'], s=data['s'],
        t=data['t'], u=data['u'], v=data['v'], w=data['w'], x=data['x'], y=data['y'],
        z=data['z'], a1=data['a1'], b1=data['b1'], c1=data['c1'], d1=data['d1']
    )
    return jsonify({'prediction': result})

# Endpoint for Heart Disease Prediction
@app.route('/predict/heartdisease', methods=['POST'])
def predict_heart_disease():
    data = request.json
    result = predictive_system_heart(
        age=data['age'], sex=data['sex'], cp=data['cp'], trestbps=data['trestbps'], chol=data['chol'],
        fbs=data['fbs'], restecg=data['restecg'], thalach=data['thalach'], exang=data['exang'],
        oldpeak=data['oldpeak'], slope=data['slope'], ca=data['ca'], thal=data['thal']
    )
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
