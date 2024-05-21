from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Load the form data
        age = float(request.form["age"])
        bmi = float(request.form["bmi"])
        HbA1c_level = float(request.form["HbA1c_level"])
        blood_glucose_level = float(request.form["blood_glucose_level"])
        gender = request.form["gender"]
        smoking_history = request.form["smoking_history"]
        hypertension = int(request.form["hypertension"])
        heart_disease = int(request.form["heart_disease"])
        
        # Create a DataFrame from the form data
        data = {
            "age": [age],
            "hypertension": [hypertension],
            "heart_disease": [heart_disease],
            "bmi": [bmi],
            "HbA1c_level": [HbA1c_level],
            "blood_glucose_level": [blood_glucose_level],
            "gender_Female": [1 if gender == "Female" else 0],
            "gender_Male": [1 if gender == "Male" else 0],
            "smoking_history_current": [1 if smoking_history == "current" else 0],
            "smoking_history_ever": [1 if smoking_history == "ever" else 0],
            "smoking_history_former": [1 if smoking_history == "former" else 0],
            "smoking_history_never": [1 if smoking_history == "never" else 0],
            "smoking_history_not current": [1 if smoking_history == "not current" else 0],
            
        }
        df = pd.DataFrame(data)
        numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']

        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

       
        model = joblib.load("trained_model.pkl")
       
        prediction = model.predict(df)
        
        return render_template("predict_result.html", prediction=prediction[0])


@app.route("/train", methods=["POST"])
def train_model():
    # Load the dataset
    df = pd.read_csv('Dataset8.csv')

    # Preprocess the data
    # Handling missing values
    age_median = df['age'].median()
    glucose_median = df['blood_glucose_level'].median()
    df['age'].fillna(age_median, inplace=True)
    df['blood_glucose_level'].fillna(glucose_median, inplace=True)
    df['smoking_history'].replace('No Info', np.nan, inplace=True)
    smoking_history_mode = df['smoking_history'].mode()[0]
    df['smoking_history'].fillna(smoking_history_mode, inplace=True)
    df['gender'].replace('Other', np.random.choice(['Male', 'Female'], p=[0.5, 0.5]), inplace=True)
    
    # Feature engineering, encoding, etc.
    df = pd.get_dummies(df, columns=['gender', 'smoking_history'])
    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    features = df.drop(columns=['diabetes'])
    target = df['diabetes']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)

    # Save the trained model
    joblib.dump(model, "trained_model.pkl")

    # Render the train result template
    return render_template("train_result.html", train_accuracy=train_accuracy, test_accuracy=test_accuracy, message="Model trained and saved successfully")



if __name__ == "__main__":
    app.run(debug=True,port=5050)
