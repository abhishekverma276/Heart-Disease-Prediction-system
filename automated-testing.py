import pickle
import numpy as np
import pandas as pd

# Load the trained Random Forest model
with open('tuned_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Feature mean and standard deviation from preprocessing step
means = {'Age': 54.0, 'BP': 130.0, 'Cholesterol': 200.0, 'Max HR': 150.0, 'ST depression': 1.0}
stds = {'Age': 9.0, 'BP': 15.0, 'Cholesterol': 40.0, 'Max HR': 20.0, 'ST depression': 1.2}

# Define test cases: list of dictionaries with inputs and expected results
test_cases = [
    # Edge Case 1: Minimum values
    {
        "input": {'Age': 20, 'cp': 1, 'BP': 90, 'Cholesterol': 100, 'Max HR': 60, 'ST depression': 0.0,
                  'thal': 3, 'chest pain type': 1, 'exercise angina': 0, 'ST slope': 1},
        "expected": "low risk"
    },
    # Edge Case 2: Maximum values
    {
        "input": {'Age': 80, 'cp': 4, 'BP': 200, 'Cholesterol': 600, 'Max HR': 220, 'ST depression': 6.5,
                  'thal': 7, 'chest pain type': 4, 'exercise angina': 1, 'ST slope': 3},
        "expected": "high risk"
    },
    # Real-world Scenario 1: Middle-aged person with mild hypertension
    {
        "input": {'Age': 45, 'cp': 2, 'BP': 140, 'Cholesterol': 220, 'Max HR': 120, 'ST depression': 1.8,
                  'thal': 3, 'chest pain type': 2, 'exercise angina': 0, 'ST slope': 2},
        "expected": "moderate risk"
    },
    # Real-world Scenario 2: Obese elderly individual
    {
        "input": {'Age': 70, 'cp': 4, 'BP': 160, 'Cholesterol': 280, 'Max HR': 90, 'ST depression': 3.5,
                  'thal': 6, 'chest pain type': 4, 'exercise angina': 1, 'ST slope': 3},
        "expected": "high risk"
    },
    # Modern-Day Athlete: Young adult with excellent health
    {
        "input": {'Age': 22, 'cp': 1, 'BP': 120, 'Cholesterol': 180, 'Max HR': 200, 'ST depression': 0.2,
                  'thal': 3, 'chest pain type': 1, 'exercise angina': 0, 'ST slope': 1},
        "expected": "low risk"
    },
    # High-Stress Corporate Worker: Chronic stress with mild symptoms
    {
        "input": {'Age': 48, 'cp': 3, 'BP': 145, 'Cholesterol': 280, 'Max HR': 130, 'ST depression': 2.5,
                  'thal': 6, 'chest pain type': 3, 'exercise angina': 1, 'ST slope': 3},
        "expected": "moderate risk"
    },
]

# Function to standardize inputs
def standardize_input(value, feature_name):
    return (value - means[feature_name]) / stds[feature_name]

# Function to run tests
def run_tests():
    for i, case in enumerate(test_cases, start=1):
        user_input = case['input']

        # Standardize the inputs
        try:
            inputs_standardized = np.array([
                standardize_input(user_input['Age'], 'Age'),
                user_input['cp'],
                standardize_input(user_input['BP'], 'BP'),
                standardize_input(user_input['Cholesterol'], 'Cholesterol'),
                standardize_input(user_input['Max HR'], 'Max HR'),
                standardize_input(user_input['ST depression'], 'ST depression'),
                user_input['thal'],
                user_input['chest pain type'],
                user_input['exercise angina'],
                user_input['ST slope']
            ]).reshape(1, -1)
        except KeyError as e:
            print(f"Test Case {i} Failed: Missing feature in input - {e}")
            print("-" * 50)
            continue

        # Predict using the model
        prediction = model.predict(inputs_standardized)[0]
        probability = model.predict_proba(inputs_standardized)[0][1]

        # Determine result
        result = "high risk" if prediction == 1 else "low risk"

        # Log test case result
        print(f"Test Case {i}:")
        print(f"Input: {user_input}")
        print(f"Expected: {case['expected']}, Predicted: {result}, Probability: {probability:.2f}")
        print("PASS" if result == case['expected'] else "FAIL")
        print("-" * 50)

# Run the tests
run_tests()
