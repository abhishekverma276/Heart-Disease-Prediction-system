Heart Disease Prediction System
Overview
The Heart Disease Prediction System is a machine learning-powered web application designed to predict the likelihood of heart disease based on user-provided health metrics. It also provides interactive exploratory data analysis (EDA) tools for visualizing dataset trends. Built with Python, the system leverages a Random Forest Classifier for accurate predictions and is deployed via Streamlit for an intuitive user interface.

Features
Heart Disease Prediction:

Users input health data, including age, blood pressure, cholesterol, and more.

The app predicts the risk level (low or high) and provides the probability of heart disease.

Exploratory Data Analysis (EDA):

Interactive visualizations allow users to explore the dataset:

Feature distributions (histograms).

Target variable analysis (bar plots for class distribution).

Scatter plots (relationships between features and target).

Correlation heatmaps (relationships between features).

Automated Testing:

A robust test suite validates model accuracy across diverse cases, such as edge cases, real-world scenarios, and invalid inputs.

Technologies Used
Programming Language: Python

Frameworks: Streamlit for web deployment

Libraries:

Data Processing: pandas, numpy

Visualization: matplotlib, seaborn

Machine Learning: scikit-learn

Setup Instructions
To set up the project, follow these steps:

Clone the Repository:

bash
git clone https://github.com/your-repository-name/heart-disease-prediction-system.git
cd heart-disease-prediction-system
Create a Virtual Environment (Optional):

bash
# For Windows:
python -m venv venv
.\venv\Scripts\activate

# For Mac/Linux:
python3 -m venv venv
source venv/bin/activate
Install Dependencies:

bash
pip install -r requirements.txt
Start the Application:

bash
streamlit run app.py
Run Automated Tests (Optional):

bash
python automated_testing.py
Folder Structure
heart-disease-prediction-system/
│
├── Data/
│   └── preprocessed_heart_data.csv        # Preprocessed dataset
│
├── app.py                                 # Streamlit app
├── EDA.py                                 # EDA script for data analysis
├── automated_testing.py                   # Automated test cases
├── tuned_rf_model.pkl                     # Trained Random Forest model
├── README.md                              # Project documentation
├── requirements.txt                       # Python dependencies
└── ...                                    # Additional files as needed
How to Use the Application
Prediction System:

Open the app in a web browser (http://localhost:8501) after running the command:

bash
streamlit run app.py
Navigate to the "Prediction System" mode.

Input health metrics such as age, cholesterol, and maximum heart rate.

Click "Predict" to view the risk level and probability of heart disease.

EDA Tools:

Switch to "Exploratory Data Analysis" mode from the sidebar.

Select the desired analysis type (e.g., feature distributions, scatter plots, heatmap).

Explore insights interactively via visualizations.

Exploratory Data Analysis Summary
Feature Distributions:

Age: Standardized values range from -2.8 to 2.49.

BP: Wide range indicates variability (-7.2 to 3.91).

Cholesterol: Outliers observed (>6.14).

Target Variable:

Balanced dataset: 53% positive cases (heart disease), 47% negative cases.

Key Relationships:

Cholesterol vs. Max HR: Weak positive correlation.

ST Depression vs. Target: Strong correlation indicates predictive importance.

Testing Details
Total Test Cases: 6

PASS: 4

FAIL: 2 (borderline cases requiring recalibration).

Results:

The model accurately predicts most cases but needs better handling of moderate risk levels.

Future Enhancements
Deploy Online:

Host the app on platforms like Heroku, AWS, or Azure.

Model Interpretability:

Use SHAP (SHapley Additive exPlanations) to explain predictions.

Improved Testing:

Add more diverse and realistic scenarios to the test suite.
