Heart Disease Prediction System
A machine learning-based web application to predict the likelihood of heart disease and explore dataset insights interactively.

Overview
The Heart Disease Prediction System is a comprehensive tool that combines machine learning predictions with exploratory data analysis (EDA). Built using a Random Forest Classifier, the project predicts the probability of heart disease based on user-provided health metrics and allows users to explore the dataset interactively through a Streamlit app.

Features
Prediction System:

Users provide health metrics (e.g., age, blood pressure, cholesterol levels).

The model predicts the heart disease risk level (low or high) with a probability score.

Exploratory Data Analysis (EDA):

Visualize feature distributions (e.g., age, blood pressure).

Analyze class balance of the target variable (heart disease: 0 or 1).

Explore relationships between features using scatter plots.

Understand feature correlations through a heatmap.

Automated Testing:

Robust test suite evaluates the model on edge cases, real-world scenarios, and invalid inputs.

Technologies Used
Programming Language: Python

Web Framework: Streamlit

Libraries:

Data Processing: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

Project Structure
heart-disease-prediction-system/
│
├── Data/
│   └── preprocessed_heart_data.csv       # Cleaned and preprocessed dataset
│
├── app.py                                # Main Streamlit app
├── EDA.py                                # EDA script for exploratory analysis
├── automated_testing.py                  # Automated test cases
├── tuned_rf_model.pkl                    # Trained Random Forest model
├── requirements.txt                      # Python dependencies
├── README.md                             # Project documentation
└── ...                                   # Add other files as needed
