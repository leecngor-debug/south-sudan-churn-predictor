from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request

BASE = Path(__file__).resolve().parent
app = Flask(__name__)
model = joblib.load(BASE / 'models' / 'best_churn_model.joblib')
preprocessor = joblib.load(BASE / 'models' / 'preprocessing_pipeline.joblib')

FIELDS = ['gender','SeniorCitizen','Partner','Dependents','tenure','PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','TotalCharges']

def risk_level(p):
    if p < 0.40: return 'Low', 'low'
    if p < 0.70: return 'Medium', 'medium'
    return 'High', 'high'

@app.route('/', methods=['GET','POST'])
def index():
    result = None
    values = {}
    error = None
    if request.method == 'POST':
        try:
            values = {f: request.form.get(f, '').strip() for f in FIELDS}
            row = values.copy()
            row['SeniorCitizen'] = int(row['SeniorCitizen'])
            tenure = int(row['tenure'])
            monthly = Decimal(row['MonthlyCharges'])
            if not 0 <= tenure <= 120 or not monthly.is_finite() or monthly < 0:
                raise ValueError('Enter a valid tenure and monthly charge.')
            total = (monthly * tenure).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
            row['tenure'] = tenure
            row['MonthlyCharges'] = float(monthly)
            row['TotalCharges'] = float(total)
            values['TotalCharges'] = f'{total:.2f}'
            X = pd.DataFrame([row], columns=FIELDS)
            Xt = preprocessor.transform(X)
            prob = float(model.predict_proba(Xt)[0,1])
            prediction = 'Likely to Churn' if prob >= 0.5 else 'Unlikely to Churn'
            risk, risk_class = risk_level(prob)
            result = {'prediction': prediction, 'probability': round(prob*100,1), 'risk': risk, 'risk_class': risk_class, 'model': 'Random Forest'}
        except (ValueError, InvalidOperation, KeyError, TypeError):
            error = 'Please enter a valid tenure and monthly charge, then try again.'
    return render_template('index.html', result=result, values=values, error=error)

@app.get('/health')
def health():
    return {'status':'ok'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
