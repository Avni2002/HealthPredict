from flask import render_template, request, Blueprint, redirect, url_for
import pickle
import numpy as np
import os

asthma_bp = Blueprint('asthma', __name__, 
                      template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

# Load trained model
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'asthma_model.pkl'), 'rb') as file:
    model = pickle.load(file)

@asthma_bp.route('/', methods=['GET', 'POST'])
def asthma_predict():
    if request.method == 'POST':
        try:
            features = [
                float(request.form['age']),
                float(request.form['gender']),               # 0 = Male, 1 = Female
                float(request.form['smoking']),              # 0 = No, 1 = Yes
                float(request.form['dust_exposure']),        # 0 = No, 1 = Yes
                float(request.form['family_history']),       # 0 = No, 1 = Yes
                float(request.form['allergies']),            # 0 = No, 1 = Yes
                float(request.form['fev1']),                 # FEV1 value
                float(request.form['fvc']),                  # FVC value
                float(request.form['wheezing']),
                float(request.form['shortness_of_breath']),
                float(request.form['chest_tightness']),
                float(request.form['coughing']),
                float(request.form['nighttime_symptoms']),
                float(request.form['exercise_induced']),
            ]
            features_array = np.array(features).reshape(1, -1)
            pred = model.predict(features_array)[0]
            prediction = 'High Risk of Asthma' if pred == 1 else 'Low Risk of Asthma'
            return redirect(url_for('asthma.asthma_result', prediction=prediction))
        except Exception as e:
            return redirect(url_for('asthma.asthma_result', prediction=f"Error: {str(e)}"))

    return render_template('asthma.html')

@asthma_bp.route('/result')
def asthma_result():
    prediction = request.args.get('prediction', 'No result found.')
    return render_template('asthma_result.html', prediction=prediction)
