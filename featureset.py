import pandas as pd

# Load the dataset used for training
data = pd.read_csv(r'selected_features_heart_data.csv')
features = data.drop(columns=['target']).columns.tolist()
print("Features expected by the model:", features)
