import pandas as pd

# Load your files
symptom_df = pd.read_csv(r"C:\Users\Meher\OneDrive\Desktop\notes\pythonprojects\Disease_prediction\Disease-Prediction-using-Machine-Learning\Symptoms_checker\dataset\dataset.csv")  # main symptom-to-disease file
desc_df = pd.read_csv(r"C:\Users\Meher\OneDrive\Desktop\notes\pythonprojects\Disease_prediction\Disease-Prediction-using-Machine-Learning\Symptoms_checker\dataset\symptom_Description.csv")
precaution_df = pd.read_csv(r"C:\Users\Meher\OneDrive\Desktop\notes\pythonprojects\Disease_prediction\Disease-Prediction-using-Machine-Learning\Symptoms_checker\dataset\symptom_precaution.csv")

# Rename columns if necessary for consistency
desc_df.columns = ['Disease', 'Description']
precaution_df.columns = ['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']

# Merge description
merged_df = symptom_df.merge(desc_df, on='Disease', how='left')

# Merge precautions
merged_df = merged_df.merge(precaution_df, on='Disease', how='left')

# Combine all precautions into a single string column
merged_df['Precautions'] = merged_df[['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].fillna('').agg(', '.join, axis=1)

# Optional: drop the individual precaution columns
merged_df.drop(columns=['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'], inplace=True)

# Save cleaned dataset
merged_df.to_csv('cleaned_symptom_dataset.csv', index=False)
print("✅ Cleaned dataset saved as 'cleaned_symptom_dataset.csv'")
