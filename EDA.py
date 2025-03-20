# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
data = pd.read_csv(r'Data/preprocessed_heart_data.csv')

# Function to print summaries and create visualizations
def print_and_plot_distribution(column):
    print(f"\nAnalyzing '{column}':")
    print(f"Mean: {data[column].mean():.2f}, Median: {data[column].median():.2f}, Std Dev: {data[column].std():.2f}")
    print(f"Min: {data[column].min()}, Max: {data[column].max()}, Null Values: {data[column].isnull().sum()}")
    
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

# 1. Distribution of Numerical Features
numerical_columns = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']
for col in numerical_columns:
    print_and_plot_distribution(col)

# 2. Target Variable Analysis
print("\nAnalyzing Target Variable ('target'):")
print(data['target'].value_counts())
print("Percentage Distribution:")
print(data['target'].value_counts(normalize=True) * 100)

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=data, palette='viridis')
plt.title('Target Variable Distribution (Heart Disease)')
plt.xlabel('Heart Disease (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.show()

# 3. Scatter Plots for Relationships
def print_and_plot_scatter(x_col, y_col):
    print(f"\nAnalyzing relationship between '{x_col}' and '{y_col}':")
    print(data[[x_col, y_col]].corr())

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=x_col, y=y_col, hue='target', data=data, palette='coolwarm')
    plt.title(f'{x_col} vs. {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Heart Disease')
    plt.show()

# Scatter plots for key relationships
print_and_plot_scatter('Cholesterol', 'Max HR')
print_and_plot_scatter('Age', 'ST depression')

# 4. Correlation Heatmap
print("\nCorrelation Matrix:")
correlation_matrix = data.corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
