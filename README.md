# Msc-Research-Project-Fraud-Detection-in-Healthcare-System

<img width="2880" height="1696" alt="image" src="https://github.com/user-attachments/assets/94e3cb2d-3c79-434b-be4c-f9f2a56591f3" />
<img width="2880" height="1696" alt="image" src="https://github.com/user-attachments/assets/1c224432-2f8a-4746-8422-1c5900af829f" />

This project is developed as part of my research work during my Masters in Data Science at Nottingham Trent University

The Healthcare Fraud Detection System is a machine learning and Flask-based dashboard application designed to identify fraudulent healthcare claims. Fraudulent billing is a major challenge in healthcare, and this project combines ensemble-based machine learning with an interactive web interface to improve fraud detection accuracy, interpretability, and usability.

The project applies a structured workflow starting with data preprocessing, exploratory analysis, and feature engineering (e.g., patient age, hospitalization duration). Class imbalance was addressed using SMOTE, and multiple algorithms—including LightGBM, XGBoost, Random Forest, Logistic Regression, and SVM—were trained and evaluated. An ensemble voting classifier (LightGBM + XGBoost + Random Forest) was developed, achieving higher recall and F1-score compared to individual models.

Evaluation metrics included accuracy, precision, recall, F1-score, ROC AUC, and PR AUC, supported by confusion matrices, ROC and precision–recall curves, and SHAP-based feature importance. The ensemble achieved strong results, with ROC AUC above 94% and PR AUC above 91%, ensuring fewer missed fraud cases while maintaining precision.

The system was deployed as a modular Flask dashboard that supports real-time predictions, batch analysis, and visualization of results. Features include a searchable dataset preview, dynamic metric displays, and explainability through SHAP plots. UI testing confirmed functionality and responsiveness.

While the system performed effectively, limitations include execution time for ensembles, dataset generalisability, and limited usability testing with fraud investigators. Future work should explore multi-source datasets, cost-sensitive learning, real-time streaming architectures, and enhanced explainability with LIME or counterfactuals.

This project demonstrates how ensemble learning and transparent dashboards can support analysts and investigators in tackling healthcare fraud more effectively.

