# Importing required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Step 1: Load Data
print("\nLoading the selected features dataset...")
data = pd.read_csv('selected_features_heart_data.csv')
X = data.drop(columns=['target'])  # Features
y = data['target']  # Target variable

# Step 2: Handle Missing Values
print("\nHandling missing values in the features (X)...")
imputer = SimpleImputer(strategy='mean')  # Replace missing values with column means
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
print("Missing values handled successfully.")

# Step 3: Split Dataset
print("\nSplitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Step 4: Train and Evaluate Models
def train_and_evaluate_model(model, model_name):
    print(f"\nTraining and evaluating {model_name}...")
    model.fit(X_train, y_train)  # Train the model
    y_pred = model.predict(X_test)  # Predict on test set
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Performance Metrics for {model_name}:")
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.title(f"ROC Curve for {model_name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.show()

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=42)
train_and_evaluate_model(logreg, "Logistic Regression")

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)
train_and_evaluate_model(decision_tree, "Decision Tree Classifier")

# Random Forest Classifier
random_forest = RandomForestClassifier(random_state=42)
train_and_evaluate_model(random_forest, "Random Forest Classifier")

# Gradient Boosting Classifier
gradient_boosting = GradientBoostingClassifier(random_state=42)
train_and_evaluate_model(gradient_boosting, "Gradient Boosting Classifier")

# (Optional) Save the best model for deployment
import pickle
best_model = random_forest  # Example: Replace with the best-performing model
print("\nSaving the best model to 'best_model.pkl'...")
with open('best_model.pkl', 'wb') as model_file:
    pickle.dump(best_model, model_file)
print("Model saved successfully!")
