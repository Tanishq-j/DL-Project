# Customer Churn Analytics & Retention Suite

An advanced, clinical-grade Customer Churn Prediction and Diagnostic Dashboard built using a **Multi-Layer Perceptron (MLP) Neural Network**. The project is designed to help subscription-based businesses (e.g. telecom, SaaS) proactively identify customers at risk of churn, score their churn probability, and output automated, actionable retention strategies.

This project is fully self-contained and serves as a **Deep Learning Lab Course Project**, strictly based on the architecture, preprocessors, and parameters from **Lab Assignment 3 (Customer Churn Prediction)**.

---

## 🚀 Key Features

* **Real-Time Customer Diagnostics**: Input customer traits (tenure, contract type, online security features, payment methods, monthly charges) and run MLP model inferences to calculate churn probability instantly.
* **Actionable Retention Playbooks**: Automatically generates customized, risk-tiered retention playbook cards ("Low Risk", "Medium Risk", "High Risk") based on the predicted churn risk.
* **Dynamic Performance Visualization**: Renders live training and validation metrics (Loss and Accuracy curves across 50 epochs) dynamically using **Chart.js** from saved neural network history.
* **Slate Noir Developer Aesthetics**: A beautiful, highly functional single-page user interface designed under strict developer-centric UI principles (Slate/Indigo monochrome color palette, clean borders, strict typography grids).

---

## 🧠 Model Architecture & Hyperparameters

The neural network is built with the following parameters matching the Lab 3 assignment:

* **Model Type**: Feedforward Multi-Layer Perceptron (MLP Classifier)
* **Input Layer**: 19 customer features (scaled using `StandardScaler` and encoded via `LabelEncoder`).
* **Hidden Layer 1**: 32 neurons, **Sigmoid (logistic)** activation function.
* **Hidden Layer 2**: 16 neurons, **Sigmoid (logistic)** activation function.
* **Output Layer**: 1 neuron, **Sigmoid (logistic)** activation (outputs churn probability).
* **Optimization Solver**: **Adam**
* **Training Epochs**: 50
* **Evaluation Accuracy**: **81.33%** on test split.

---

## 📁 Repository Structure

```directory
e:/DL course project/
│
├── Assignments/           # Lab assignments (PDFs and Notebooks)
│
├── Project/               # Selected Course Project folder
│   ├── generate_data.py   # Synthesizes realistic churn.csv telecom data
│   ├── train_model.py     # Data pipeline, encoding, and neural network training script
│   ├── app.py             # Lightweight Flask backend server & inference API
│   │
│   ├── templates/         # UI templates
│   │   └── index.html     # High-fidelity dashboard, forms, and Chart.js layout
│   │
│   # --- Serialized Model Artifacts (Generated after training) ---
│   ├── churn.csv          # Generated customer dataset (1,500 samples)
│   ├── churn_model.pkl    # Trained MLP Neural Network weights
│   ├── scaler.pkl         # Fitted StandardScaler state
│   ├── label_encoders.pkl # Fitted LabelEncoders dictionary 
│   ├── encoder_mappings.pkl # Classes mapping for dropdown population
│   ├── history.pkl        # Serialized loss & accuracy training curves
│   └── feature_names.pkl  # Exact feature column order
│
└── Readme.md              # Documentation and execution guide (This file)
```

---

## 🛠️ Step-by-Step Installation & Run Guide

The project is completely self-contained and pre-trained, ready to run with these simple steps:

### 1. Set Up Your Environment
Ensure you have the required dependencies installed on your system:
```bash
pip install pandas numpy scikit-learn Flask
```
*(Note: Scikit-Learn is used to run the neural network `MLPClassifier` to ensure a bulletproof CPU-based runtime on Windows without heavy PyTorch or TensorFlow downloads).*

### 2. Generate Dataset & Train Model (Optional)
If you want to regenerate the data or retrain the neural network from scratch, run the scripts in the following order:
```bash
# Generate the synthetic telecom churn CSV
python Project/generate_data.py

# Train the Sigmoid MLP model & save artifacts
python Project/train_model.py
```

### 3. Launch the Portal
Start the Flask web application backend:
```bash
python Project/app.py
```

Once running, open your web browser and navigate to:
👉 **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

## 📊 Verification & Metrics

Upon running `train_model.py`, the training outputs loss updates across epochs:
```txt
Stochastic Optimizer: Maximum iterations (50) reached.
Training neural network...
Iteration 1, loss = 0.70224543
...
Iteration 25, loss = 0.60581364
...
Iteration 50, loss = 0.46330624

Final Test Accuracy: 81.33%
Model saved successfully to Project/churn_model.pkl
```
The dashboard dynamically fetches these loss and accuracy points from `history.pkl` and plots the dynamic learning curves on the **Model Performance** page.
