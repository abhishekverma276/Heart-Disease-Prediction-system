# Importing required libraries for feature selection
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Load the preprocessed dataset
print("\nLoading the dataset...")
data = pd.read_csv('Data/preprocessed_heart_data.csv')

# Step 1: Check for Missing Values in Target Variable
print("\nStep 1: Checking for missing values in the 'target' column...")
if data['target'].isnull().sum() > 0:
    print(f"Found {data['target'].isnull().sum()} missing values in 'target'. Dropping rows with missing target values.")
    data = data.dropna(subset=['target'])
else:
    print("No missing values in the 'target' column.")

# Step 2: Separate Features and Target Variable
print("\nStep 2: Separating features (X) and target variable (y)...")
X = data.drop(columns=['target'])  # All features except the target
y = data['target']  # Target variable (Heart Disease)

# Step 3: Remove Features with Only NaN Values
print("\nStep 3: Removing columns with only NaN values...")
nan_columns = X.columns[X.isnull().all()]
if len(nan_columns) > 0:
    print(f"Found columns with only NaN values: {list(nan_columns)}. Dropping these columns.")
    X = X.drop(columns=nan_columns)
else:
    print("No columns with only NaN values.")

# Step 4: Handle Missing Values in Features
print("\nStep 4: Handling missing values in features (X)...")
imputer = SimpleImputer(strategy='mean')  # Replace NaNs with column means
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Missing values handled by imputing with column means.")

# Step 5: Correlation Filtering
print("\nStep 5: Performing correlation filtering...")
correlation_with_target = data.corr()['target'].sort_values(ascending=False)
print("Top Correlated Features with Target:\n", correlation_with_target.head(10))

# Step 6: Recursive Feature Elimination (RFE)
print("\nStep 6: Performing Recursive Feature Elimination (RFE)...")
# Using RandomForestClassifier as the base model
model = RandomForestClassifier(random_state=42)
rfe = RFE(model, n_features_to_select=10)  # Select the top 10 features
rfe.fit(X, y)
selected_features_rfe = X.columns[rfe.support_]
print("Selected Features by RFE:\n", selected_features_rfe)

# Step 7: Principal Component Analysis (PCA)
print("\nStep 7: Performing Principal Component Analysis (PCA)...")
pca = PCA(n_components=5)  # Reduce to 5 principal components
X_pca = pca.fit_transform(X)
print("Explained Variance Ratio by PCA:\n", pca.explained_variance_ratio_)
print("Shape of the PCA-reduced dataset:", X_pca.shape)

# Step 8: Save Selected Features
print("\nStep 8: Saving selected features for further modeling...")
selected_features_data = data[selected_features_rfe]
selected_features_data['target'] = y  # Add the target back to the selected feature set
selected_features_data.to_csv('selected_features_heart_data.csv', index=False)
print("Selected features saved to 'selected_features_heart_data.csv'.")
