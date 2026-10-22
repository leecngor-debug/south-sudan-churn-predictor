from pathlib import Path
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
import os
import io

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_file

BASE = Path(__file__).resolve().parent

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024

model = joblib.load(BASE / 'models' / 'best_churn_model.joblib')
preprocessor = joblib.load(BASE / 'models' / 'preprocessing_pipeline.joblib')

FIELDS = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges'
]


def risk_level(p):
    if p < 0.40:
        return 'Low', 'low'
    if p < 0.70:
        return 'Medium', 'medium'
    return 'High', 'high'


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    values = {}
    error = None

    if request.method == 'POST':
        try:
            values = {
                f: request.form.get(f, '').strip()
                for f in FIELDS
            }

            row = values.copy()

            row['SeniorCitizen'] = int(row['SeniorCitizen'])

            tenure = int(row['tenure'])
            monthly = Decimal(row['MonthlyCharges'])

            if (
                not 0 <= tenure <= 120
                or not monthly.is_finite()
                or monthly < 0
            ):
                raise ValueError(
                    'Enter a valid tenure and monthly charge.'
                )

            total = (monthly * tenure).quantize(
                Decimal('0.01'),
                rounding=ROUND_HALF_UP
            )

            row['tenure'] = tenure
            row['MonthlyCharges'] = float(monthly)
            row['TotalCharges'] = float(total)

            values['TotalCharges'] = f'{total:.2f}'

            X = pd.DataFrame([row], columns=FIELDS)
            Xt = preprocessor.transform(X)

            prob = float(model.predict_proba(Xt)[0, 1])

            prediction = (
                'Likely to Churn'
                if prob >= 0.5
                else 'Unlikely to Churn'
            )

            risk, risk_class = risk_level(prob)

            result = {
                'prediction': prediction,
                'probability': round(prob * 100, 1),
                'risk': risk,
                'risk_class': risk_class,
                'model': 'Random Forest'
            }

        except (
            ValueError,
            InvalidOperation,
            KeyError,
            TypeError
        ):
            error = (
                'Please enter a valid tenure and monthly charge, '
                'then try again.'
            )

    return render_template(
        'index.html',
        result=result,
        values=values,
        error=error
    )


@app.post('/batch')
def batch():
    error = None
    batch_results = None
    results_csv = None

    try:
        uploaded = request.files.get('file')

        if uploaded is None or uploaded.filename == '':
            raise ValueError('Please choose a CSV file.')

        if not uploaded.filename.lower().endswith('.csv'):
            raise ValueError('Please upload a CSV file.')

        df = pd.read_csv(uploaded)

        if len(df) == 0:
            raise ValueError('The CSV file is empty.')

        if len(df) > 1000:
            raise ValueError(
                'Please upload no more than 1,000 customers.'
            )

        missing = [field for field in FIELDS if field not in df.columns]

        if missing:
            raise ValueError(
                'Missing required columns: ' + ', '.join(missing)
            )

        X = df[FIELDS].copy()

        X['SeniorCitizen'] = pd.to_numeric(
            X['SeniorCitizen'],
            errors='raise'
        ).astype(int)

        X['tenure'] = pd.to_numeric(
            X['tenure'],
            errors='raise'
        ).astype(int)

        X['MonthlyCharges'] = pd.to_numeric(
            X['MonthlyCharges'],
            errors='raise'
        )

        X['TotalCharges'] = pd.to_numeric(
            X['TotalCharges'],
            errors='coerce'
        )

        missing_total = X['TotalCharges'].isna()

        X.loc[missing_total, 'TotalCharges'] = (
            X.loc[missing_total, 'MonthlyCharges']
            * X.loc[missing_total, 'tenure']
        )

        Xt = preprocessor.transform(X)

        probabilities = model.predict_proba(Xt)[:, 1]

        output = df.copy()

        output['ChurnProbability'] = [
            round(float(p) * 100, 1)
            for p in probabilities
        ]

        output['Prediction'] = [
            'Likely to Churn'
            if p >= 0.5
            else 'Unlikely to Churn'
            for p in probabilities
        ]

        output['RiskLevel'] = [
            risk_level(float(p))[0]
            for p in probabilities
        ]

        batch_results = output.to_dict(orient='records')
        results_csv = output.to_csv(index=False)

    except Exception as exc:
        error = str(exc)

    return render_template(
        'index.html',
        result=None,
        values={},
        error=error,
        batch_results=batch_results,
        results_csv=results_csv
    )


@app.get('/download')
def sample():
    return send_file(
        BASE / 'sample_customers.csv',
        as_attachment=True,
        download_name='sample_customers.csv'
    )


@app.get('/health')
def health():
    return {'status': 'ok'}, 200


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000))
    )