import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

# Load data
heart_data = pd.read_csv(r'C:\Users\JAI AHUJA\Desktop\SIH\AI\heart_disease_data.csv')

# Splitting the data
X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Accuracy for training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(y_train, X_train_prediction)  # Correct order of arguments

# Accuracy for test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(y_test, X_test_prediction)  # Correct order of arguments



def predictive_system_heart(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape numpy array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Predict
    predictions = model.predict(input_data_reshaped)
    if predictions[0] == 0:
        print("Healthy Heart")
    else:
        print("Have a Heart Disease")

predictive_system_heart(65,1,0,110,248,0,0,158,0,0.6,2,2,1)
# Save the model
with open('heart_model.pkl', 'wb') as f:
    pickle.dump(model, f)