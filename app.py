from pathlib import Path
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
            row['tenure'] = float(row['tenure'])
            row['MonthlyCharges'] = float(row['MonthlyCharges'])
            row['TotalCharges'] = float(row['TotalCharges'])
            if row['tenure'] < 0 or row['MonthlyCharges'] < 0 or row['TotalCharges'] < 0:
                raise ValueError('Numeric values cannot be negative.')
            X = pd.DataFrame([row], columns=FIELDS)
            Xt = preprocessor.transform(X)
            prob = float(model.predict_proba(Xt)[0,1])
            prediction = 'Likely to Churn' if prob >= 0.5 else 'Unlikely to Churn'
            risk, risk_class = risk_level(prob)
            result = {'prediction': prediction, 'probability': round(prob*100,1), 'risk': risk, 'risk_class': risk_class, 'model': 'Random Forest'}
        except Exception as exc:
            error = f'Please check the customer details and try again. ({exc})'
    return render_template('index.html', result=result, values=values, error=error)
@app.route('/bulk', methods=['POST'])
def bulk_predict():
    try:
        file = request.files.get('file')

        if not file or file.filename == '':
            return render_template(
                'index.html',
                bulk_error='Please select a CSV file.'
            )

        if not file.filename.lower().endswith('.csv'):
            return render_template(
                'index.html',
                bulk_error='Please upload a CSV file.'
            )

        df = pd.read_csv(file)

        # Make sure all model fields are available
        missing = [field for field in FIELDS if field not in df.columns]

        # TotalCharges can be calculated automatically
        if 'TotalCharges' in missing:
            if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
                df['TotalCharges'] = (
                    pd.to_numeric(df['tenure'], errors='coerce') *
                    pd.to_numeric(df['MonthlyCharges'], errors='coerce')
                )
                missing.remove('TotalCharges')

        if missing:
            raise ValueError(
                'Missing required columns: ' + ', '.join(missing)
            )

        # Convert numeric fields
        df['SeniorCitizen'] = pd.to_numeric(
            df['SeniorCitizen'], errors='raise'
        ).astype(int)

        df['tenure'] = pd.to_numeric(
            df['tenure'], errors='raise'
        )

        df['MonthlyCharges'] = pd.to_numeric(
            df['MonthlyCharges'], errors='raise'
        )

        df['TotalCharges'] = pd.to_numeric(
            df['TotalCharges'], errors='coerce'
        )

        # Automatically fill missing TotalCharges
        calculated_total = df['tenure'] * df['MonthlyCharges']
        df['TotalCharges'] = df['TotalCharges'].fillna(calculated_total)

        # Run all customers through the existing model
        X = df[FIELDS].copy()
        Xt = preprocessor.transform(X)

        probabilities = model.predict_proba(Xt)[:, 1]

        df['ChurnProbability'] = (probabilities * 100).round(1)
        df['Prediction'] = [
            'Likely to Churn' if p >= 0.5
            else 'Unlikely to Churn'
            for p in probabilities
        ]

        df['RiskLevel'] = [
            risk_level(float(p))[0]
            for p in probabilities
        ]

        # Highest-risk customers appear first
        df = df.sort_values(
            'ChurnProbability',
            ascending=False
        )

        bulk_results = df.to_dict(orient='records')

        return render_template(
            'index.html',
            bulk_results=bulk_results,
            bulk_count=len(df)
        )

    except Exception as exc:
        return render_template(
            'index.html',
            bulk_error=f'Could not process the file: {exc}'
        )
@app.get('/health')
def health():
    return {'status':'ok'}, 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
