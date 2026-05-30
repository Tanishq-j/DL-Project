import pandas as pd
import numpy as np
import os

def generate_telecom_churn_data(file_path, num_samples=1500):
    np.random.seed(42)
    
    print("Generating synthetic but realistic telecom churn dataset...")
    
    # Feature distributions
    genders = np.random.choice(['Male', 'Female'], size=num_samples)
    senior_citizens = np.random.choice([0, 1], p=[0.84, 0.16], size=num_samples)
    partners = np.random.choice(['Yes', 'No'], p=[0.48, 0.52], size=num_samples)
    dependents = np.random.choice(['Yes', 'No'], p=[0.30, 0.70], size=num_samples)
    
    # Tenure has a bi-modal distribution (new customers & very loyal customers)
    tenures = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        if np.random.rand() < 0.4:
            tenures[i] = np.random.randint(1, 12)
        elif np.random.rand() < 0.3:
            tenures[i] = np.random.randint(12, 48)
        else:
            tenures[i] = np.random.randint(48, 73)
            
    phone_service = np.random.choice(['Yes', 'No'], p=[0.90, 0.10], size=num_samples)
    
    multiple_lines = []
    for i in range(num_samples):
        if phone_service[i] == 'No':
            multiple_lines.append('No phone service')
        else:
            multiple_lines.append(np.random.choice(['Yes', 'No'], p=[0.45, 0.55]))
            
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], p=[0.34, 0.44, 0.22], size=num_samples)
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for i in range(num_samples):
        if internet_services[i] == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No'], p=[0.35, 0.65]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.40, 0.60]))
            device_protection.append(np.random.choice(['Yes', 'No'], p=[0.41, 0.59]))
            tech_support.append(np.random.choice(['Yes', 'No'], p=[0.36, 0.64]))
            streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.49, 0.51]))
            streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.50, 0.50]))
            
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], p=[0.55, 0.21, 0.24], size=num_samples)
    paperless_billing = np.random.choice(['Yes', 'No'], p=[0.59, 0.41], size=num_samples)
    
    payment_methods = np.random.choice([
        'Electronic check', 'Mailed check', 
        'Bank transfer (automatic)', 'Credit card (automatic)'
    ], p=[0.34, 0.23, 0.21, 0.22], size=num_samples)
    
    # Calculate realistic charges
    monthly_charges = []
    total_charges = []
    
    for i in range(num_samples):
        # Base fee
        base = 20.0
        if phone_service[i] == 'Yes':
            base += 10.0
        if multiple_lines[i] == 'Yes':
            base += 15.0
        if internet_services[i] == 'DSL':
            base += 30.0
        elif internet_services[i] == 'Fiber optic':
            base += 45.0
            
        if internet_services[i] != 'No':
            if online_security[i] == 'Yes': base += 8.0
            if online_backup[i] == 'Yes': base += 8.0
            if device_protection[i] == 'Yes': base += 8.0
            if tech_support[i] == 'Yes': base += 8.0
            if streaming_tv[i] == 'Yes': base += 12.0
            if streaming_movies[i] == 'Yes': base += 12.0
            
        # Add a tiny bit of noise to monthly charges
        monthly = base + np.random.normal(0, 3)
        monthly = max(18.0, round(monthly, 2))
        monthly_charges.append(monthly)
        
        # Total charges = Monthly * Tenure
        total = round(monthly * tenures[i], 2)
        total_charges.append(total)
        
    # Generate predictive Churn with realistic correlations
    # high risk: month-to-month contract, no tech support, fiber optic, high charges, low tenure
    churn_list = []
    for i in range(num_samples):
        score = 0.0
        # Tenure effect (longer tenure = less churn)
        score -= (tenures[i] / 12.0) * 0.8
        
        # Contract type (month-to-month has high risk)
        if contracts[i] == 'Month-to-month':
            score += 1.5
        elif contracts[i] == 'Two year':
            score -= 1.0
            
        # Internet service type (fiber optic has high churn in real datasets due to pricing/support issues)
        if internet_services[i] == 'Fiber optic':
            score += 0.6
            
        # Tech support & security (decreases risk)
        if internet_services[i] != 'No':
            if tech_support[i] == 'No':
                score += 0.5
            if online_security[i] == 'No':
                score += 0.4
                
        # Payment method (electronic check has higher churn)
        if payment_methods[i] == 'Electronic check':
            score += 0.5
            
        # Monthly charges (higher charges = more churn)
        if monthly_charges[i] > 70:
            score += 0.4
            
        # Base churn offset
        score -= 0.5
        
        # Convert to probability
        prob = 1.0 / (1.0 + np.exp(-score))
        
        # Decide Churn (Yes/No)
        if np.random.rand() < prob:
            churn_list.append('Yes')
        else:
            churn_list.append('No')
            
    # Customer ID
    customer_ids = [f"{np.random.randint(1000, 9999)}-{np.random.choice(['A','B','C','D','E'])}{np.random.choice(['W','X','Y','Z'])}{np.random.randint(10,99)}" for _ in range(num_samples)]
    
    # Build dataframe
    df = pd.DataFrame({
        'customerID': customer_ids,
        'gender': genders,
        'SeniorCitizen': senior_citizens,
        'Partner': partners,
        'Dependents': dependents,
        'tenure': tenures,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_services,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contracts,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_methods,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Churn': churn_list
    })
    
    # Introduce some missing values or string format in TotalCharges to match the lab cleanup
    # Make a few TotalCharges values empty strings (representing newly joined customers with 0 tenure)
    df.loc[df['tenure'] == 0, 'TotalCharges'] = ' '
    
    # Ensure folder structure
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"Dataset generated and saved successfully to {file_path} (Shape: {df.shape})")

if __name__ == '__main__':
    generate_telecom_churn_data('Project/churn.csv')
