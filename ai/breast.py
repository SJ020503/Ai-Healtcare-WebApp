import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import sklearn.datasets

# Load the dataset
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Create a DataFrame with the features
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

# Target variable
target = breast_cancer_dataset.target

# Features and target
X = data_frame
y = target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy on training data
train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)

# Accuracy on test data
test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_pred)



def predictive_system_breast(a,b,c,d,e,f,g,h,i,j,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,b1,c1,d1):
    input_data = (a,b,c,d,e,f,g,h,i,j,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,a1,b1,c1,d1)
    # Convert input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    
    # Reshape the numpy array as we are predicting for one data point
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Predict
    prediction = model.predict(input_data_reshaped)
    
    # Interpret the result
    if prediction[0] == 0:
        print('The Breast Cancer is Malignant')
    else:
        print('The Breast Cancer is Benign')


# Save the model
with open('breast_model.pkl', 'wb') as f:
    pickle.dump(model, f)