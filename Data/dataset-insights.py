# Importing necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dynamically locating file paths
base_dir = os.path.join(os.path.dirname(os.getcwd()), 'Data')
file1 = os.path.join(base_dir, 'dataset.csv')
file2 = os.path.join(base_dir, 'Heart_Disease_Prediction.csv')
file3 = os.path.join(base_dir, 'heart_statlog_cleveland_hungary_final.csv')

# Loading datasets with validation
try:
    dataset1 = pd.read_csv(file1)
    dataset2 = pd.read_csv(file2)
    dataset3 = pd.read_csv(file3)
except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure the datasets exist at the specified path.")
    exit()

# Function to explore each dataset
def dataset_summary(df, name):
    print(f"\n{'='*40}\nDataset: {name}\n{'='*40}")
    print("Shape of the dataset:", df.shape)
    print("\nColumns and data types:\n", df.dtypes)
    print("\nFirst few rows of data:\n", df.head())
    print("\nMissing values per column:\n", df.isnull().sum())
    print("\nBasic statistics:\n", df.describe())

# Analyze each dataset
dataset_summary(dataset1, "Dataset 1")
dataset_summary(dataset2, "Dataset 2")
dataset_summary(dataset3, "Dataset 3")

# Visualizing missing values for all datasets
plt.figure(figsize=(15, 5))
sns.heatmap(dataset1.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Dataset 1")
plt.show()

sns.heatmap(dataset2.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Dataset 2")
plt.show()

sns.heatmap(dataset3.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values in Dataset 3")
plt.show()

# Function to find correlations and plot heatmap
def correlation_heatmap(df, name):
    print(f"\nCorrelation Matrix for {name}:\n")
    correlation = df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Correlation Heatmap for {name}")
    plt.show()

# Plot correlation heatmap for each dataset
correlation_heatmap(dataset1, "Dataset 1")
correlation_heatmap(dataset2, "Dataset 2")
correlation_heatmap(dataset3, "Dataset 3")

# Extracting Key Insights
def extract_insights(df, name):
    print(f"\n{'-'*40}\nInsights for {name}:\n{'-'*40}")
    print(f"Total missing values: {df.isnull().sum().sum()}")
    print(f"Top correlated features with target variable (if applicable):")
    if 'target' in df.columns:
        print(df.corr()['target'].sort_values(ascending=False)[1:6])

extract_insights(dataset1, "Dataset 1")
extract_insights(dataset2, "Dataset 2")
extract_insights(dataset3, "Dataset 3")
