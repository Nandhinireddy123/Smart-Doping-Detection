import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset from CSV file
file_path = "balanced_anti_doping_dataset.csv"
df = pd.read_csv(file_path)

# Add 'Doping_Type' column with random categorical values
doping_types = ['Steroids', 'EPO', 'Stimulants', 'Diuretics', 'HGH']
np.random.seed(42)
df['Doping_Type'] = np.random.choice(doping_types, size=len(df))

# Print first few rows instead of display
print("First few rows of the dataset:")
print(df.head())

# Print dataset summary
print("\nDataset summary:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# List available columns
print("\nAvailable columns in dataset:")
print(df.columns)

# Ensure 'Test_Results' exists before using it as the target
if 'Test_Results' not in df.columns:
    raise KeyError("'Test_Results' column is missing from the dataset!")

# Define features and target for doping detection
X = df.drop(columns=['Test_Results', 'Athlete_ID', 'Doping_Type'], errors='ignore')
y = df['Test_Results']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature Importance
feature_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,5))
feature_importances.plot(kind='bar', color='skyblue')
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Classification")
plt.show()

# Predicting the type of doping consumed
X_doping = df.drop(columns=['Doping_Type', 'Athlete_ID'], errors='ignore')
y_doping = df['Doping_Type']

X_train_doping, X_test_doping, y_train_doping, y_test_doping = train_test_split(X_doping, y_doping, test_size=0.2, random_state=42)

doping_clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
doping_clf.fit(X_train_doping, y_train_doping)

y_pred_doping = doping_clf.predict(X_test_doping)

# Classification Report for Doping Type Prediction
print("\nDoping Type Classification Report:")
print(classification_report(y_test_doping, y_pred_doping))

# Confusion Matrix for Doping Type Prediction
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test_doping, y_pred_doping), annot=True, fmt='d', cmap='Oranges', xticklabels=doping_types, yticklabels=doping_types)
plt.xlabel("Predicted Doping Type")
plt.ylabel("Actual Doping Type")
plt.title("Confusion Matrix for Doping Type Prediction")
plt.show()

# Save the models for later use
import pickle
# Save the doping detection model
with open('doping_detection_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

# Save the doping type classification model
with open('doping_type_model.pkl', 'wb') as f:
    pickle.dump(doping_clf, f)

print("\nModels have been saved as 'doping_detection_model.pkl' and 'doping_type_model.pkl'")

# Function to predict drug type based on input athlete data
def predict_doping_type(input_data):
    input_df = pd.DataFrame([input_data], columns=X_doping.columns)
    predicted_doping_type = doping_clf.predict(input_df)[0]
    return predicted_doping_type

# Example athlete data input
for i in range(0,10):
    example_athlete = X_doping.iloc[i].to_dict()
    predicted_drug = predict_doping_type(example_athlete)
    print(f"\nPredicted Doping Type for the given athlete data: {predicted_drug}")

def load_models():
    with open('doping_detection_model.pkl', 'rb') as f:
        detection_model = pickle.load(f)
    with open('doping_type_model.pkl', 'rb') as f:
        type_model = pickle.load(f)
    return detection_model, type_model
