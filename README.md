# South Sudan Churn Predictor

Web-based ML customer churn prediction prototype for telecommunication services in South Sudan.

## Features
- Individual customer churn prediction
- Churn probability percentage
- Low / Medium / High risk level
- Random Forest model selected from LR, DT, RF and SVM comparison
- Input validation and responsive interface

## Run locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:5000`.

## Deploy on Render
1. Create a GitHub repository and upload this project.
2. In Render choose **New > Blueprint** and connect the repository, or create a Web Service manually.
3. The included `render.yaml` requests the service name `south-sudan-churn-predictor`.
4. If that name is available, the expected URL is `https://south-sudan-churn-predictor.onrender.com`.
5. If unavailable, choose a close alternative such as `south-sudan-telecom-churn`.

The exact free subdomain is assigned by Render and depends on name availability.
