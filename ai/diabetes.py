#Importing libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle


diabetes_dataset = pd.read_csv(r'C:\Users\JAI AHUJA\Desktop\SIH\AI\diabetes.csv')
#0 --> Non-Diabetic          1 --> Diabetic
X = diabetes_dataset.drop(columns='Outcome',axis=1)
y = diabetes_dataset['Outcome']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=2)
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train,y_train)

#accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction , y_train)

#accuracy of the test data
X_test_predictions = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predictions , y_test)

def pdf_reader():
    #Predictive System
    import pdfplumber
    import pandas as pd
    import re

    # Define the path to the PDF
    pdf_path = "/content/TestDiabetes.pdf"

    # Initialize a list to store the extracted data
    data = []

    # Define a function to extract numeric values from a line of text
    def extract_value(label, text):
        pattern = rf"{label}\s*:\s*([\d\.]+)"
        match = re.search(pattern, text)
        return float(match.group(1)) if match else None

    # Open and read the PDF
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()

            # Extract features (if not present, they'll be None)
            pregnancies = extract_value('Pregnancies', text)
            glucose = extract_value('Glucose', text)
            blood_pressure = extract_value('BloodPressure', text)
            skin_thickness = extract_value('SkinThickness', text)
            insulin = extract_value('Insulin', text)
            bmi = extract_value('BMI', text)
            diabetes_pedigree = extract_value('DiabetesPedigreeFunction', text)
            age = extract_value('Age', text)

            # Store the extracted data in a dictionary (None if missing)
            data.append({
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': diabetes_pedigree,
                'Age': age
            })

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(data)

    # Handle missing data by filling with default values or using other imputation methods
    default_values = {
        'Pregnancies': 0,
        'Glucose': df['Glucose'].mean() if 'Glucose' in df else 100,  # Mean or default
        'BloodPressure': df['BloodPressure'].mean() if 'BloodPressure' in df else 70,
        'SkinThickness': df['SkinThickness'].mean() if 'SkinThickness' in df else 20,
        'Insulin': df['Insulin'].mean() if 'Insulin' in df else 80,
        'BMI': df['BMI'].mean() if 'BMI' in df else 25.0,
        'DiabetesPedigreeFunction': df['DiabetesPedigreeFunction'].mean() if 'DiabetesPedigreeFunction' in df else 0.5,
        'Age': df['Age'].mean() if 'Age' in df else 40
    }

    # Fill NaN values with default values
    df_filled = df.fillna(value=default_values)

    # Display the filled DataFrame
    print(df_filled)

    # Drop rows with any missing values
    df_clean = df.dropna()

    # Ensure there's at least one row remaining after dropping NaNs
    if df_clean.empty:
        print("No valid data available for prediction.")
    else:
        # Select the row you want to use for prediction, e.g., the first row (index 0)
        selected_row = df_clean.iloc[0]

        # Extract features from the selected row and convert to tuple or list
        input_data = (
            selected_row['Pregnancies'],
            selected_row['Glucose'],
            selected_row['BloodPressure'],
            selected_row['SkinThickness'],
            selected_row['Insulin'],
            selected_row['BMI'],
            selected_row['DiabetesPedigreeFunction'],
            selected_row['Age']
        )

        # Convert input data into a numpy array
        input_data_as_numpy_array = np.asarray(input_data)

        # Reshape numpy array as we are predicting for only one instance
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Assuming `classifier` is your trained model, make the prediction
        predictions = classifier.predict(input_data_reshaped)

        # Interpret the prediction
        if predictions[0] == 0:
            print("No Signs of Diabetes")
        else:
            print("Have Diabetes")


#Building a predictive system withouit pdf
def predictive_system_diabetes(Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age):
    input_data = (Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age)
    input_data_as_numpy_array = np.asarray(input_data)

    #reshaping numpy array as we are prediction for only one data
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    predictions = classifier.predict(input_data_reshaped)
    if(predictions[0]==0):
        print("No Signs of Diabetes")
    else:
        print("Have Diabetes")

predictive_system_diabetes(5,166,72,19,175,25.8,0.587,51)

with open('diabetes_model.pkl','wb') as f:
    pickle.dump(classifier,f)



    