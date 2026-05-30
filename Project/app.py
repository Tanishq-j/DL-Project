from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load all trained model artifacts and encoders
print("Loading model artifacts...")
with open('Project/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('Project/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('Project/label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('Project/categorical_columns.pkl', 'rb') as f:
    categorical_columns = pickle.load(f)

with open('Project/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

with open('Project/encoder_mappings.pkl', 'rb') as f:
    encoder_mappings = pickle.load(f)

with open('Project/mean_total_charges.pkl', 'rb') as f:
    mean_total_charges = pickle.load(f)

with open('Project/history.pkl', 'rb') as f:
    training_history = pickle.load(f)

print("Artifacts loaded successfully!")

@app.route('/')
def home():
    # Pass encoder mappings to populate select dropdowns on the frontend dynamically
    return render_template('index.html', mappings=encoder_mappings)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from POST request (form or JSON)
        if request.is_json:
            form_data = request.get_json()
        else:
            form_data = request.form.to_dict()
            
        print("Received form data:", form_data)
        
        # Prepare a dictionary for processing
        input_data = {}
        
        # Process numeric features
        try:
            input_data['tenure'] = float(form_data.get('tenure', 0))
        except ValueError:
            input_data['tenure'] = 0.0
            
        try:
            input_data['SeniorCitizen'] = int(form_data.get('SeniorCitizen', 0))
        except ValueError:
            input_data['SeniorCitizen'] = 0
            
        try:
            input_data['MonthlyCharges'] = float(form_data.get('MonthlyCharges', 0))
        except ValueError:
            input_data['MonthlyCharges'] = 0.0
            
        try:
            total_charges_raw = form_data.get('TotalCharges', '').strip()
            if not total_charges_raw:
                input_data['TotalCharges'] = mean_total_charges
            else:
                input_data['TotalCharges'] = float(total_charges_raw)
        except ValueError:
            input_data['TotalCharges'] = mean_total_charges
            
        # Process categorical features using the fitted label encoders
        for col in categorical_columns:
            val = form_data.get(col, '')
            le = label_encoders.get(col)
            if le is not None:
                # If value is missing or not in the classes, fallback to the most common class (index 0)
                if not val or val not in le.classes_:
                    input_data[col] = 0
                else:
                    input_data[col] = int(le.transform([val])[0])
            else:
                input_data[col] = 0
                
        # Reorder features exactly as feature_names.pkl
        ordered_features = []
        for name in feature_names:
            ordered_features.append(input_data[name])
            
        # Convert to numpy array and scale
        features_arr = np.array(ordered_features).reshape(1, -1)
        scaled_features = scaler.transform(features_arr)
        
        # Run prediction
        churn_probability = float(model.predict_proba(scaled_features)[0][1])
        churn_risk = round(churn_probability * 100, 2)
        
        # Determine risk category & actionable recommendation
        if churn_risk < 30.0:
            category = "Low Risk"
            color = "success"
            recommendations = [
                "Maintain standard service quality.",
                "Customer has high loyalty; suitable for cross-selling premium add-ons.",
                "Conduct routine satisfaction checkups periodically."
            ]
        elif churn_risk < 70.0:
            category = "Medium Risk"
            color = "warning"
            recommendations = [
                "Send a proactive check-in email from the customer support team.",
                "Offer a small discount on their current plan or a free 1-month add-on.",
                "Recommend transitioning to an annual contract if they are on a Month-to-Month plan."
            ]
        else:
            category = "High Risk"
            color = "danger"
            recommendations = [
                "ACTION REQUIRED: Immediate outreach by account manager is recommended.",
                "Offer a highly incentivized 1-year contract extension or loyalty credit.",
                "Resolve any active support tickets or billing disputes immediately."
            ]
            
        return jsonify({
            'success': True,
            'probability': churn_probability,
            'risk_percentage': churn_risk,
            'category': category,
            'color': color,
            'recommendations': recommendations
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/history-data')
def history_data():
    # Serve training history for dynamic Chart.js rendering on frontend
    return jsonify(training_history)

if __name__ == '__main__':
    print("Starting Customer Churn Flask server...")
    app.run(debug=True, port=5000)
