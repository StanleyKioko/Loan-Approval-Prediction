from flask import Flask, request, render_template
import pandas as pd
import joblib

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = joblib.load('loan_approval_model.pkl')

# Define a function to preprocess the input data
def preprocess_data(data):
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']
    
    # Convert input data to a DataFrame
    df = pd.DataFrame([data])
    
    # Ensure the columns in the input data match those expected by the model
    for col in categorical_columns:
        if col not in df.columns:
            df[col] = ['']
    
    # Perform one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Ensure the columns match the training data
    expected_columns = model.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[expected_columns]
    
    return df

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Extract form data
    data = {
        'Gender': request.form.get('Gender'),
        'Married': request.form.get('Married'),
        'Dependents': request.form.get('Dependents'),
        'Education': request.form.get('Education'),
        'Self_Employed': request.form.get('Self_Employed'),
        'Property_Area': request.form.get('Property_Area'),
        'ApplicantIncome': float(request.form.get('ApplicantIncome')),
        'CoapplicantIncome': float(request.form.get('CoapplicantIncome')),
        'LoanAmount': float(request.form.get('LoanAmount')),
        'Loan_Amount_Term': float(request.form.get('Loan_Amount_Term')),
        'Credit_History': float(request.form.get('Credit_History'))
    }
    
    # Preprocess the input data
    processed_data = preprocess_data(data)
    
    # Make a prediction
    prediction = model.predict(processed_data)
    prediction_proba = model.predict_proba(processed_data)
    
    # Determine the prediction result
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    probability = prediction_proba[0][1] * 100  # Convert to percentage
    
    return render_template('result.html', result=result, probability=probability)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
