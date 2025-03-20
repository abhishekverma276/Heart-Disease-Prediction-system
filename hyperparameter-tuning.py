# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Load the dataset
print("\nLoading the selected features dataset...")
data = pd.read_csv('selected_features_heart_data.csv')
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Handle missing values (if any)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Split data into training and testing sets
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],            # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],          # Maximum depth of trees
    'min_samples_split': [2, 5, 10],          # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],            # Minimum number of samples required to be a leaf node
    'bootstrap': [True, False]                # Whether bootstrap samples are used when building trees
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Perform Grid Search Cross Validation
print("\nPerforming Grid Search for Hyperparameter Tuning...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best hyperparameters and their corresponding model
print("\nBest Hyperparameters Found:")
print(grid_search.best_params_)

# Evaluate the tuned model on the test set
best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test)
y_prob = best_rf.predict_proba(X_test)[:, 1]

print("\nPerformance of Tuned Random Forest Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the tuned model
import pickle
print("\nSaving the tuned Random Forest model to 'tuned_rf_model.pkl'...")
with open('tuned_rf_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf, model_file)
print("Tuned model saved successfully!")
