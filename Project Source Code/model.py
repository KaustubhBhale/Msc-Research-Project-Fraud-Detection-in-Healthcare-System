import os
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from joblib import dump, load
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, average_precision_score
)


def load_options(filename):
    """Load possible options for categorical fields"""
    path = os.path.join('static', 'input_fields', filename)
    try:
        with open(path, 'r') as f:
            return sorted(set(line.strip() for line in f if line.strip()))
    except FileNotFoundError:
        return []


# Load dataset
df = pd.read_csv('data/fraud_model_data_new.csv')

# Target encoding
df['PotentialFraud'] = df['PotentialFraud'].map({'Yes': 1, 'No': 0})

# Features and target
X = df.drop(columns=['PotentialFraud'])
y = df['PotentialFraud']

# High cardinality columns
high_card_cols = ['Provider', 'AttendingPhysician', 'OperatingPhysician',
                  'OtherPhysician', 'State', 'County']

# Load known category options
all_categories = {
    'Provider': load_options('providers.txt'),
    'AttendingPhysician': load_options('attending_physicians.txt'),
    'OperatingPhysician': load_options('operating_physicians.txt'),
    'OtherPhysician': load_options('other_physicians.txt'),
    'State': load_options('states.txt'),
    'County': load_options('counties.txt')
}

# Label encoding
label_encoders = {}
for col in high_card_cols:
    le = LabelEncoder()
    all_values = list(X[col].unique()) + all_categories.get(col, [])
    le.fit(all_values)
    X[col] = le.transform(X[col].astype(str))
    label_encoders[col] = le

# One-hot encode low-cardinality columns
low_card_cols = [col for col in X.select_dtypes(include='object') if col not in high_card_cols]
X = pd.get_dummies(X, columns=low_card_cols)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Define models
lgb_model = lgb.LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=100)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42)

# Ensemble voting classifier
voting = VotingClassifier(estimators=[
    ('lgb', lgb_model),
    ('rf', rf_model),
    ('xgb', xgb_model)
], voting='soft')

# Train ensemble
voting.fit(X_train, y_train)

# Evaluate model
ensemble_pred = voting.predict(X_test)
ensemble_proba = voting.predict_proba(X_test)[:, 1]

#print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))
#print("\nClassification Report:\n", classification_report(y_test, ensemble_pred))

# Optional: print confusion matrix and AUCs (text only)
#print("Confusion Matrix:\n", confusion_matrix(y_test, ensemble_pred))
#print("ROC AUC Score:", roc_auc_score(y_test, ensemble_proba))
#print("Average Precision Score:", average_precision_score(y_test, ensemble_proba))

# Save model artifacts
feature_columns = X.columns.tolist()
dump({
    'model': voting,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'high_card_cols': high_card_cols,
    'low_card_cols': low_card_cols
}, 'fraud_model.joblib')


# ----------------------------
# Prediction Function
# ----------------------------
def predict_fraud_model(input_data: dict):
    artifacts = load('fraud_model.joblib')
    model = artifacts['model']
    label_encoders = artifacts['label_encoders']
    feature_columns = artifacts['feature_columns']
    high_card_cols = artifacts['high_card_cols']
    low_card_cols = artifacts['low_card_cols']

    input_df = pd.DataFrame([input_data])

    for col in high_card_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
            input_df[col] = le.transform(input_df[col].astype(str))

    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]
    return pred, proba
