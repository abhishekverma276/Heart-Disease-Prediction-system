import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the preprocessed dataset for EDA
data = pd.read_csv(r'Data/preprocessed_heart_data.csv')

# Load the trained Random Forest model
with open('tuned_rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Feature mean and standard deviation from preprocessing step
means = {'Age': 54.0, 'BP': 130.0, 'Cholesterol': 200.0, 'Max HR': 150.0, 'ST depression': 1.0}
stds = {'Age': 9.0, 'BP': 15.0, 'Cholesterol': 40.0, 'Max HR': 20.0, 'ST depression': 1.2}

# App title and description
st.title("Heart Disease Prediction System")
st.write("""
This app predicts the likelihood of heart disease based on health metrics and allows you to explore the dataset through visualizations. Enter your data below for an instant prediction or switch to EDA to understand trends in the dataset.
""")

# Sidebar for navigation
st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Prediction System", "Exploratory Data Analysis"])

if app_mode == "Prediction System":
    # User inputs for real-world values
    age = st.number_input("Age (Years)", 20, 80, step=1)  # Real-world Age
    cp = st.selectbox("Chest Pain Type (CP)", ["1: Typical Angina", "2: Atypical Angina", 
                                               "3: Non-anginal Pain", "4: Asymptomatic"])
    cp = int(cp.split(":")[0])

    bp = st.number_input("Blood Pressure (BP, mmHg)", 90, 200, step=1)  # Real-world BP
    cholesterol = st.number_input("Cholesterol Level (mg/dL)", 100, 600, step=1)  # Real-world Cholesterol
    max_hr = st.number_input("Maximum Heart Rate Achieved (bpm)", 60, 220, step=1)  # Real-world Max HR
    st_depression = st.number_input("ST Depression", 0.0, 6.5, step=0.1)  # Real-world ST Depression

    thal = st.selectbox("Thalassemia (Thal)", ["3: Normal", "6: Fixed Defect", "7: Reversible Defect"])
    thal = int(thal.split(":")[0])

    chest_pain_type = st.selectbox("Chest Pain Type (Detailed)", ["1: Typical", "2: Atypical", 
                                                                  "3: Non-anginal", "4: Asymptomatic"])
    chest_pain_type = int(chest_pain_type.split(":")[0])

    exercise_angina = st.selectbox("Exercise-Induced Angina", ["0: No", "1: Yes"])
    exercise_angina = int(exercise_angina.split(":")[0])

    st_slope = st.selectbox("ST Slope", ["1: Upsloping", "2: Flat", "3: Downsloping"])
    st_slope = int(st_slope.split(":")[0])

    # Standardize the inputs
    age_std = (age - means['Age']) / stds['Age']
    bp_std = (bp - means['BP']) / stds['BP']
    cholesterol_std = (cholesterol - means['Cholesterol']) / stds['Cholesterol']
    max_hr_std = (max_hr - means['Max HR']) / stds['Max HR']
    st_depression_std = (st_depression - means['ST depression']) / stds['ST depression']

    # Prepare input for the model
    user_input = np.array([[age_std, cp, bp_std, cholesterol_std, max_hr_std, st_depression_std, thal, 
                            chest_pain_type, exercise_angina, st_slope]])

    # Validate input shape
    st.write(f"Input shape: {user_input.shape[1]} (Expected: 10)")

    # Prediction
    if st.button("Predict"):
        try:
            prediction = model.predict(user_input)
            probability = model.predict_proba(user_input)[0][1]
            if prediction[0] == 1:
                st.error(f"The model predicts a high risk of heart disease with a probability of {probability:.2f}. Please consult a healthcare provider.")
            else:
                st.success(f"The model predicts a low risk of heart disease with a probability of {probability:.2f}.")
        except ValueError as e:
            st.error(f"Error: {e}. Please verify your inputs.")

elif app_mode == "Exploratory Data Analysis":
    # EDA Section
    eda_option = st.sidebar.selectbox("Choose EDA Analysis", [
        "Feature Summaries", "Target Analysis", "Scatter Plot Relationships", "Correlation Matrix"
    ])

    # Feature Summaries
    if eda_option == "Feature Summaries":
        st.header("Feature Summaries for Numerical Features")
        numerical_columns = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
        feature = st.selectbox("Select a Feature to Analyze", numerical_columns)
        st.write(f"**Analyzing '{feature}':**")
        st.write(f"Mean: {data[feature].mean():.2f}, Median: {data[feature].median():.2f}, Std Dev: {data[feature].std():.2f}")
        st.write(f"Min: {data[feature].min()}, Max: {data[feature].max()}, Null Values: {data[feature].isnull().sum()}")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(data[feature], kde=True, bins=20, color='skyblue', ax=ax)
        ax.set_title(f'Distribution of {feature}', fontsize=16)
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)

    # Target Analysis
    elif eda_option == "Target Analysis":
        st.header("Target Variable Analysis ('target')")
        st.write(data['target'].value_counts())
        st.write("**Percentage Distribution:**")
        st.write(data['target'].value_counts(normalize=True) * 100)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='target', data=data, palette='viridis', ax=ax)
        ax.set_title('Target Variable Distribution (Heart Disease)', fontsize=16)
        ax.set_xlabel('Heart Disease (0 = No, 1 = Yes)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)

    # Scatter Plot Relationships
    elif eda_option == "Scatter Plot Relationships":
        st.header("Scatter Plot Relationships")
        x_col = st.selectbox("Select X-axis Feature", ['Cholesterol', 'Age'])
        y_col = st.selectbox("Select Y-axis Feature", ['Max HR', 'ST depression'])
        st.write(f"Correlation between '{x_col}' and '{y_col}':")
        st.write(data[[x_col, y_col]].corr())
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.scatterplot(x=x_col, y=y_col, hue='target', data=data, palette='coolwarm', ax=ax)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=16)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.legend(title='Heart Disease')
        st.pyplot(fig)

    # Correlation Matrix
    elif eda_option == "Correlation Matrix":
        st.header("Correlation Matrix")
        st.write("Correlation Matrix:")
        st.write(data.corr())
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=16)
        st.pyplot(fig)
