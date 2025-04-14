import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
import joblib
import os
os.makedirs('models', exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("cleaned_symptom_dataset.csv")

# 1. Collect symptom columns (auto-detect)
symptom_cols = [col for col in df.columns if col.lower().startswith('symptom')]

# 2. Fill missing symptoms with ''
df[symptom_cols] = df[symptom_cols].fillna('')

# 3. Combine all symptoms into a list per row
df['SymptomList'] = df[symptom_cols].values.tolist()

# 4. Create a unique symptom set
all_symptoms = sorted(set(s for row in df['SymptomList'] for s in row if s))

# 5. Convert symptom lists to binary vectors
X = pd.DataFrame([[1 if s in row else 0 for s in all_symptoms] for row in df['SymptomList']], columns=all_symptoms)

# 6. Encode diseases
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Disease'])

# 7. Train model
model = MultinomialNB()
model.fit(X, y)

# 8. Save model and assets
joblib.dump(model, 'models/symptom_checker_model.pkl')
joblib.dump(all_symptoms, 'models/symptom_list.pkl')
joblib.dump(label_encoder, 'models/label_encoder.pkl')

print("✅ Model training complete! Files saved in /models")
