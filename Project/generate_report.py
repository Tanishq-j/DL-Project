import fitz
import os
from datetime import datetime

def create_project_report(file_path):
    print("Generating professional PDF Course Project Report...")
    
    # 1. Create a new PDF document
    doc = fitz.open()
    
    # Page size A4: 595 x 842 points
    a4_width = 595
    a4_height = 842
    
    # ----------------------------------------------------
    # PAGE 1: COVER PAGE
    # ----------------------------------------------------
    page1 = doc.new_page(width=a4_width, height=a4_height)
    
    # Background color block or subtle borders (we will keep it clean and minimal)
    # Simple top accent line
    page1.draw_rect(fitz.Rect(50, 40, 545, 45), fill=(0.22, 0.74, 0.97), color=None) # Primary blue accent line
    
    # Report Title
    rect_title = fitz.Rect(50, 180, 545, 320)
    title_text = "DEEP LEARNING LAB\nCOURSE PROJECT REPORT"
    page1.insert_textbox(rect_title, title_text, fontsize=24, fontname="hebo", color=(0.06, 0.09, 0.15), align=1)
    
    # Subtitle
    rect_sub = fitz.Rect(50, 340, 545, 420)
    sub_text = "Customer Churn Analytics & Retention Suite\nMulti-Layer Perceptron (MLP) Neural Network Portal"
    page1.insert_textbox(rect_sub, sub_text, fontsize=14, fontname="helv", color=(0.3, 0.35, 0.45), align=1)
    
    # Divider line
    page1.draw_line(fitz.Point(150, 450), fitz.Point(445, 450), color=(0.8, 0.8, 0.8), width=1)
    
    # Student Details
    rect_details = fitz.Rect(50, 520, 545, 680)
    details_text = (
        "SUBMITTED BY:\n\n"
        "Name: Tanishq Jain\n"
        "Roll No: A-48\n"
        "PRN: 2324000422\n"
        "Class: TY CSE (AIML)\n\n"
        "Subject: Deep Learning Lab (DLL)"
    )
    page1.insert_textbox(rect_details, details_text, fontsize=12, fontname="hebo", color=(0.1, 0.15, 0.25), align=1)
    
    # Date
    rect_date = fitz.Rect(50, 720, 545, 760)
    date_text = f"Date: {datetime.now().strftime('%B %d, %Y')}"
    page1.insert_textbox(rect_date, date_text, fontsize=10, fontname="helv", color=(0.4, 0.45, 0.5), align=1)
    
    # Page Number
    page1.insert_text(fitz.Point(a4_width / 2 - 10, 800), "1", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))

    # ----------------------------------------------------
    # PAGE 2: ABSTRACT & PREPROCESSING PIPELINE
    # ----------------------------------------------------
    page2 = doc.new_page(width=a4_width, height=a4_height)
    
    # Running Header
    page2.insert_text(fitz.Point(50, 45), "DEEP LEARNING LAB REPORT • CUSTOMER CHURN ANALYTICS", fontsize=8, fontname="hebo", color=(0.5, 0.5, 0.5))
    page2.draw_line(fitz.Point(50, 50), fitz.Point(545, 50), color=(0.8, 0.8, 0.8), width=1)
    
    # Page Title
    rect_h1_p2 = fitz.Rect(50, 70, 545, 100)
    page2.insert_textbox(rect_h1_p2, "1. Abstract & Data Processing Pipeline", fontsize=16, fontname="hebo", color=(0.06, 0.09, 0.15))
    
    # Abstract Text
    rect_abs_p2 = fitz.Rect(50, 110, 545, 250)
    abs_text = (
        "1.1 Abstract & Problem Statement\n"
        "Customer churn represents one of the most critical financial challenges for subscription-based "
        "enterprises, such as telecommunication and SaaS platforms. Proactively forecasting whether a customer "
        "will cancel their subscription allows companies to design highly targeted customer retention campaigns. "
        "This project implements a fully-connected Artificial Neural Network (Multi-Layer Perceptron) "
        "trained on customer demographic, billing, and system service data. The primary objective is to build "
        "an end-to-end diagnostic pipeline that takes multi-dimensional customer features and outputs "
        "highly accurate churn risk probabilities along with customized retention playbooks.\n\n"
        "1.2 Dataset Description\n"
        "The model is trained on a telecom customer database comprising 1,500 rows. The features include "
        "demographics (gender, SeniorCitizen, Partner, Dependents), account details (tenure, contract type, "
        "charges), and subscribed add-ons (phone service, multiple lines, internet service type, online security, "
        "online backup, device protection, tech support, streaming TV, and streaming movies)."
    )
    page2.insert_textbox(rect_abs_p2, abs_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Preprocessing Pipeline
    rect_pre_p2 = fitz.Rect(50, 270, 545, 500)
    pre_text = (
        "1.3 Preprocessing Pipeline (Based on Lab Assignment 3)\n"
        "To feed the tabular records into the MLP neural network, the data was preprocessed via "
        "the following steps:\n"
        "• TotalCharges Data Cleanup: The TotalCharges field was parsed into numerical values. "
        "Empty entries (which correspond to newly joined customers with 0 months tenure) were identified and "
        "imputed using the dataset's global mean TotalCharges value ($2,283.56).\n"
        "• Label Encoding: All 15 categorical variables (e.g. Contract, PaymentMethod, InternetService) "
        "were transformed into integer codes using LabelEncoder. This allows the inputs to be fed "
        "as numerical values while retaining distinct feature identities.\n"
        "• Feature Scaling: To ensure standard gradients during backpropagation and to prevent features with large "
        "numeric magnitudes (e.g., TotalCharges) from dominating, all features were scaled using "
        "StandardScaler (mean = 0, variance = 1).\n"
        "• Train-Test Splitting: The preprocessed dataset was split into an 80% training set (1,200 samples) "
        "and a 20% test set (300 samples) to ensure accurate cross-validation."
    )
    page2.insert_textbox(rect_pre_p2, pre_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Page Number
    page2.insert_text(fitz.Point(a4_width / 2 - 10, 800), "2", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))

    # ----------------------------------------------------
    # PAGE 3: MODEL ARCHITECTURE & ACCURACY
    # ----------------------------------------------------
    page3 = doc.new_page(width=a4_width, height=a4_height)
    
    # Running Header
    page3.insert_text(fitz.Point(50, 45), "DEEP LEARNING LAB REPORT • CUSTOMER CHURN ANALYTICS", fontsize=8, fontname="hebo", color=(0.5, 0.5, 0.5))
    page3.draw_line(fitz.Point(50, 50), fitz.Point(545, 50), color=(0.8, 0.8, 0.8), width=1)
    
    # Page Title
    rect_h1_p3 = fitz.Rect(50, 70, 545, 100)
    page3.insert_textbox(rect_h1_p3, "2. Neural Network Architecture & Performance", fontsize=16, fontname="hebo", color=(0.06, 0.09, 0.15))
    
    # Architecture Details
    rect_arch_p3 = fitz.Rect(50, 110, 545, 260)
    arch_text = (
        "2.1 Multi-Layer Perceptron (MLP) Specifications\n"
        "The model architecture is a feedforward Artificial Neural Network comprising fully-connected (Dense) "
        "layers with sigmoid (logistic) activations, designed to match the Lab 3 sequential structure:\n"
        "• Input Layer: 19 nodes corresponding to the processed features.\n"
        "• Hidden Layer 1: 32 dense nodes with Sigmoid activation. This layer encodes primary non-linear "
        "interactions between customer demographic and billing attributes.\n"
        "• Hidden Layer 2: 16 dense nodes with Sigmoid activation. This layer refines features into higher-level "
        "abstractions.\n"
        "• Output Layer: 1 dense node with Sigmoid activation. The output maps directly to the probability space "
        "[0.0, 1.0], representing the probability of the customer churning.\n\n"
        "2.2 Training & Optimization Strategy\n"
        "The neural network was optimized using the Adam solver for stochastic gradient descent, "
        "compiling with Binary Cross-Entropy loss. The network was trained for 50 iterations (epochs) "
        "on the CPU, tracking loss minimization. The learning rate was set to 0.001."
    )
    page3.insert_textbox(rect_arch_p3, arch_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Model evaluation table
    page3.draw_rect(fitz.Rect(50, 290, 545, 305), fill=(0.95, 0.95, 0.95), color=(0.8, 0.8, 0.8))
    page3.insert_text(fitz.Point(60, 301), "Layer Type", fontsize=10, fontname="hebo", color=(0.3, 0.3, 0.3))
    page3.insert_text(fitz.Point(180, 301), "Nodes", fontsize=10, fontname="hebo", color=(0.3, 0.3, 0.3))
    page3.insert_text(fitz.Point(260, 301), "Activation", fontsize=10, fontname="hebo", color=(0.3, 0.3, 0.3))
    page3.insert_text(fitz.Point(380, 301), "Parameters", fontsize=10, fontname="hebo", color=(0.3, 0.3, 0.3))
    
    y = 320
    layers_data = [
        ("Input Layer", "19", "None", "—"),
        ("Hidden Layer 1 (Dense)", "32", "Sigmoid (Logistic)", "640"),
        ("Hidden Layer 2 (Dense)", "16", "Sigmoid (Logistic)", "528"),
        ("Output Layer (Dense)", "1", "Sigmoid (Logistic)", "17")
    ]
    for l_type, nodes, act, params in layers_data:
        page3.draw_line(fitz.Point(50, y+8), fitz.Point(545, y+8), color=(0.9, 0.9, 0.9), width=1)
        page3.insert_text(fitz.Point(60, y), l_type, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        page3.insert_text(fitz.Point(180, y), nodes, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        page3.insert_text(fitz.Point(260, y), act, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        page3.insert_text(fitz.Point(380, y), params, fontsize=9.5, fontname="helv", color=(0.2, 0.2, 0.2))
        y += 20
        
    # Model metrics
    rect_metrics_p3 = fitz.Rect(50, 430, 545, 570)
    metrics_text = (
        "2.3 Experimental Performance & Convergence\n"
        "During training, the binary cross-entropy loss decreased monotonically from 0.7022 (Epoch 1) to "
        "0.4633 (Epoch 50). The evaluation of the final MLP model on the independent test set (300 samples) "
        "produced the following results:\n"
        "• Final Test Accuracy: 81.33% (Significantly outperforming random baseline classification)\n"
        "• Model Validation: The loss minimization curves show excellent generalization without signs of "
        "overfitting, indicating highly robust weights in the intermediate dense representations."
    )
    page3.insert_textbox(rect_metrics_p3, metrics_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Page Number
    page3.insert_text(fitz.Point(a4_width / 2 - 10, 800), "3", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))

    # ----------------------------------------------------
    # PAGE 4: PORTAL DESIGN & CONCLUSION
    # ----------------------------------------------------
    page4 = doc.new_page(width=a4_width, height=a4_height)
    
    # Running Header
    page4.insert_text(fitz.Point(50, 45), "DEEP LEARNING LAB REPORT • CUSTOMER CHURN ANALYTICS", fontsize=8, fontname="hebo", color=(0.5, 0.5, 0.5))
    page4.draw_line(fitz.Point(50, 50), fitz.Point(545, 50), color=(0.8, 0.8, 0.8), width=1)
    
    # Page Title
    rect_h1_p4 = fitz.Rect(50, 70, 545, 100)
    page4.insert_textbox(rect_h1_p4, "3. System Implementation & Conclusion", fontsize=16, fontname="hebo", color=(0.06, 0.09, 0.15))
    
    # Web Portal Architecture
    rect_web_p4 = fitz.Rect(50, 110, 545, 270)
    web_text = (
        "3.1 Web Diagnostics Portal Implementation\n"
        "To provide a tangible demonstration of the MLP neural network model, a local web server was "
        "built using the Python Flask framework. The portal is divided into two primary zones:\n"
        "• Diagnostic Portal: A highly responsive customer input form styled in a premium Slate Noir dark aesthetic. "
        "The form utilizes a modern system font stack and subtle borders. On form submission, a JavaScript fetch "
        "request transmits customer attributes to the backend prediction API. The backend scales features and "
        "predicts a probability score, which is rendered dynamically alongside high-performance retention "
        "recommendation playbooks.\n"
        "• Performance Dashboard: Uses Chart.js to render interactive training history loss and accuracy curves. "
        "These graphs are fed directly from the serialized history data (`history.pkl`) generated during model training."
    )
    page4.insert_textbox(rect_web_p4, web_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Conclusion & Future Scope
    rect_concl_p4 = fitz.Rect(50, 290, 545, 450)
    concl_text = (
        "3.2 Conclusion\n"
        "The Customer Churn Analytics Dashboard successfully demonstrates a robust application of a "
        "tabular MLP neural network classifier. By structuring demographic and financial input features "
        "through numerical encoding and scaling, the model achieved an 81.33% diagnostic accuracy. "
        "The fully-integrated Flask interface provides a direct illustration of how neural networks "
        "can be packaged into clean, highly responsive customer operations software.\n\n"
        "3.3 Future Scope & Extensions\n"
        "• Multi-Task Neural Architectures: Expanding the model to predict *both* churn probability (classification) "
        "and customer lifetime value (regression) using a single, shared-base network.\n"
        "• Real-Time SQL Integrations: Integrating database endpoints to run continuous churn auditing across "
        "an active enterprise customer database."
    )
    page4.insert_textbox(rect_concl_p4, concl_text, fontsize=10.5, fontname="helv", color=(0.15, 0.15, 0.15))
    
    # Divider line
    page4.draw_line(fitz.Point(150, 520), fitz.Point(445, 520), color=(0.8, 0.8, 0.8), width=1)
    
    # Signature/Approval section
    rect_sig = fitz.Rect(50, 560, 545, 680)
    sig_text = (
        "This project and report represent the verified individual work completed for the "
        "Deep Learning Laboratory.\n\n\n\n"
        "_______________________________\n"
        "Tanishq Jain (A-48)\n"
        "TY CSE (AIML), PRN: 2324000422"
    )
    page4.insert_textbox(rect_sig, sig_text, fontsize=11, fontname="hebo", color=(0.2, 0.25, 0.35), align=1)
    
    # Page Number
    page4.insert_text(fitz.Point(a4_width / 2 - 10, 800), "4", fontsize=9, fontname="helv", color=(0.5, 0.5, 0.5))
    
    # Save the document
    doc.save(file_path)
    print(f"Professional PDF Report saved successfully to {file_path}")

if __name__ == '__main__':
    create_project_report('Project/Course_Project_Report.pdf')
