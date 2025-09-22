import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, \
    precision_recall_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from model import voting, X_test, y_test, ensemble_proba, X_train, y_train
import base64
from io import BytesIO
import shap
from lightgbm import LGBMClassifier


def get_model_metrics():
    """Calculate model performance metrics"""
    y_pred = voting.predict(X_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, ensemble_proba)
    }

    return metrics


def get_feature_importance():
    """Calculate feature importance from the ensemble model and return top 5"""
    feature_names = X_train.columns
    n_features = len(feature_names)
    importances = np.zeros(n_features)
    valid_estimators = 0

    # Use fitted models from named_estimators_
    for name in voting.named_estimators_:
        model = voting.named_estimators_[name]
        if hasattr(model, 'feature_importances_'):
            model_importances = model.feature_importances_
            if len(model_importances) == n_features:
                abs_importances = np.abs(model_importances)
                total_importance = np.sum(abs_importances)
                if total_importance > 0:
                    normalized_importances = abs_importances / total_importance
                    importances += normalized_importances
                    valid_estimators += 1

    # Average importances if valid models exist
    if valid_estimators > 0:
        importances /= valid_estimators

    # Create DataFrame and sort by importance
    feature_imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Always return top 5
    return feature_imp_df.head(5).reset_index(drop=True)


def get_classification_report():
    """Generate classification report as a dictionary"""
    y_pred = voting.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    return report


def plot_roc_curve():
    """Generate ROC curve plot"""
    fpr, tpr, _ = roc_curve(y_test, ensemble_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, ensemble_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic


def plot_precision_recall_curve():
    """Generate Precision-Recall curve plot"""
    precision, recall, _ = precision_recall_curve(y_test, ensemble_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic


def plot_confusion_matrix():
    """Generate confusion matrix plot"""
    y_pred = voting.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic


def plot_probability_distribution():
    """Generate probability distribution plot"""
    plt.figure(figsize=(8, 6))
    sns.histplot(ensemble_proba, bins=50, kde=True)
    plt.title('Predicted Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Count')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic


def plot_shap_summary():
    """Generate SHAP summary plot for top 5 features"""
    # Extract the LGBMClassifier from the ensemble
    lgbm_model = voting.estimators[0][1]  # Get the LGBMClassifier

    # Refit the LGBMClassifier with the training data
    lgbm_model_refit = LGBMClassifier(class_weight='balanced', random_state=42, n_estimators=100)
    lgbm_model_refit.fit(X_train, y_train)

    # Initialize SHAP explainer with the refitted model
    explainer = shap.TreeExplainer(lgbm_model_refit)
    shap_values = explainer.shap_values(X_test)

    # Get top 5 features based on feature importance
    feature_imp_df = get_feature_importance()
    top_5_features = feature_imp_df['feature'].tolist()
    feature_indices = [list(X_test.columns).index(f) for f in top_5_features]

    # Filter SHAP values and features for top 5
    shap_values_filtered = shap_values[:, feature_indices]
    X_test_filtered = X_test[top_5_features]

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values_filtered, X_test_filtered, feature_names=top_5_features, plot_type="bar")
    plt.title('SHAP Summary Plot (Top 5 Features)')

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    graphic = base64.b64encode(image_png).decode('utf-8')
    plt.close()

    return graphic
