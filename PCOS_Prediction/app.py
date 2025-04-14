from flask import render_template, request, Blueprint, redirect, url_for
import pickle
import joblib
import numpy as np
import os

# Create the PCOS blueprint
pcos_bp = Blueprint('pcos', __name__, 
                    template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Load trained model for PCOS
pcos_model = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcos_model.pkl'))

# Load the scaler for PCOS
pcos_scaler = joblib.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pcos_scaler.pkl'))
# Route for the PCOS prediction page
@pcos_bp.route('/', methods=['GET', 'POST'])
def pcos_predict():
    if request.method == 'POST':
        try:
            # Get features from the form
            features = [
                float(request.form['age']),  # Age
                float(request.form['bmi']),  # BMI
                float(request.form['menstrual_irregularity']),  # Menstrual Irregularity (binary or categorical)
                float(request.form['testosterone_level']),  # Testosterone Level
                float(request.form['antral_follicle_count'])  # Antral Follicle Count
            ]
            # Scale the features
            features_scaled = pcos_scaler.transform([features])
            
            # Make the prediction
            pred = pcos_model.predict(features_scaled)[0]
            
            # Convert the prediction to a readable result
            prediction = 'High Risk of PCOS' if pred == 1 else 'Low Risk of PCOS'
            
            return redirect(url_for('pcos.pcos_result', prediction=prediction))
        except Exception as e:
            return redirect(url_for('pcos.pcos_result', prediction=f"Error: {str(e)}"))

    return render_template('pcos.html')

# Route to display the result of the prediction
@pcos_bp.route('/result')
def pcos_result():
    prediction = request.args.get('prediction', 'No result found.')
    return render_template('pcos_result.html', prediction=prediction)
