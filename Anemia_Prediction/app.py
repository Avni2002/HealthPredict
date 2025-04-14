from flask import Blueprint, render_template, request, redirect, url_for

import numpy as np
import joblib
import os

anemia_bp = Blueprint('anemia', __name__, template_folder='templates')

# Load model, scaler, encoder
model_path = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(model_path, 'models/anemia_model.pkl'))
scaler = joblib.load(os.path.join(model_path, 'models/anemia_scaler.pkl'))
label_encoder = joblib.load(os.path.join(model_path, 'models/anemia_label_encoder.pkl'))

@anemia_bp.route('/', methods=['GET', 'POST'])
def anemia_predict():
    if request.method == 'POST':
        try:
            gender = request.form['gender'].capitalize()
            hemoglobin = float(request.form['hemoglobin'])
            mchc = float(request.form['mchc'])
            mcv = float(request.form['mcv'])
            mch = float(request.form['mch'])
            gender_encoded = 0 if gender == 'Male' else 1

            features = np.array([[gender_encoded, hemoglobin, mchc, mcv, mch]])
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            decoded = label_encoder.inverse_transform([pred])[0]
            prediction = "High Risk of Anemia" if decoded.lower() == 'anemia' else "Normal"

            # 🔁 Redirect with query string
            return redirect(url_for('anemia.anemia_result', prediction=prediction))

        except Exception as e:
            return redirect(url_for('anemia.anemia_result', prediction=f"Error: {str(e)}"))

    return render_template('anemia.html')

@anemia_bp.route('/result')
def anemia_result():
    prediction = request.args.get('prediction', 'No result available')
    return render_template('anemia_result.html', prediction=prediction)
