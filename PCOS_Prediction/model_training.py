import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Sample data based on your provided dataset
data = {
    'Age': [24, 37, 32, 28, 25],
    'BMI': [34.7, 26.4, 23.6, 28.8, 22.1],
    'Menstrual_Irregularity': [1, 0, 0, 0, 1],
    'Testosterone_Level': [25.2, 57.1, 92.7, 63.1, 59.8],
    'Antral_Follicle_Count': [20, 25, 28, 26, 8],
    'PCOS_Diagnosis': [0, 0, 0, 0, 0]  # 0 means no PCOS, 1 would indicate PCOS
}

df = pd.DataFrame(data)

# Separate features and target variable
X = df.drop('PCOS_Diagnosis', axis=1)
y = df['PCOS_Diagnosis']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(model, 'pcos_model.pkl')
joblib.dump(scaler, 'pcos_scaler.pkl')
