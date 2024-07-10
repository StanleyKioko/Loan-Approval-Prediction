import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load datasets
train = pd.read_csv(r"C:\Users\HP\Desktop\Loan Approval Prediction\Training Dataset.csv")
test = pd.read_csv(r"C:\Users\HP\Desktop\Loan Approval Prediction\Test Dataset.csv")

# Print the column names to verify they are as expected
print("Train Columns:", train.columns)
print("Test Columns:", test.columns)

# Combine train and test data for consistent preprocessing
combined = pd.concat([train, test], sort=False)

# Verify combined DataFrame columns
print("Combined Columns:", combined.columns)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# List of categorical columns based on your dataset's columns
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

# Encode categorical columns for 'Loan_Status' only in the training data
for column in categorical_columns:
    if column in combined.columns:
        if column == 'Loan_Status':
            # Fit label encoder only on the training data
            train[column] = label_encoder.fit_transform(train[column])
        else:
            # Use the same encoder to transform both train and test data
            combined[column] = label_encoder.fit_transform(combined[column])
    else:
        print(f"Column '{column}' not found in combined DataFrame")

# Separate the combined data back into train and test sets
train_encoded = combined[:len(train)]
test_encoded = combined[len(train):]

# Handle any missing values
imputer = SimpleImputer(strategy='most_frequent')
train_encoded = pd.DataFrame(imputer.fit_transform(train_encoded), columns=train_encoded.columns)
test_encoded = pd.DataFrame(imputer.transform(test_encoded), columns=test_encoded.columns)

# Features and target variable
X = train_encoded.drop(['Loan_ID', 'Loan_Status'], axis=1)  # Update columns to match the dataset
y = train_encoded['Loan_Status']  # Update the target variable to 'Loan_Status'

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
report = classification_report(y_val, y_pred)

print(f'Validation Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')

# Save the model
joblib.dump(model, 'loan_approval_model.pkl')
