# asthma_model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load your data
df = pd.read_csv("asthmadataset.csv")  # make sure this CSV exists

# Features & Target
X = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluation (optional)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
with open("asthma_model.pkl", "wb") as f:
    pickle.dump(model, f)
