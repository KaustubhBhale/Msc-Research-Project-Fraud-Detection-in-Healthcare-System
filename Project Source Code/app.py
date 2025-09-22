from flask import Flask, jsonify, render_template, request, flash

from bulk_predictor import model_bulk_predict
from data_loader import load_dataset
import pandas as pd
from flask import send_file
from werkzeug.utils import secure_filename

from model import predict_fraud_model
from model_metrics import get_model_metrics, plot_roc_curve, plot_precision_recall_curve, \
    plot_confusion_matrix, plot_probability_distribution, plot_shap_summary, \
    get_classification_report, get_feature_importance
from visualisations import get_reporting_data
from helper import load_options
from joblib import load

app = Flask(__name__)

#reading the cleaned dataset
df = pd.read_csv('data/fraud_model_data_new.csv')


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def predict():
    data, columns = load_dataset()
    return render_template('index.html', preview_table=data, column_names=columns)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/data_preview')
def data_preview():
    data, columns = load_dataset()
    # data is already limited to 30 rows
    return render_template('index.html',
                           preview_table=data,
                           column_names=columns)


@app.route('/predict_fraud', methods=['GET', 'POST'])
def predict_fraud_page():
    # Load all dropdown options
    dropdown_options = {
        'providers': load_options('providers.txt'),
        'attending_physicians': load_options('attending_physicians.txt'),
        'operating_physicians': load_options('operating_physicians.txt'),
        'other_physicians': load_options('other_physicians.txt'),
        'states': load_options('states.txt'),
        'counties': load_options('counties.txt'),
        'race_options': load_options('race_options.txt')
    }
    if request.method == 'POST':
        try:
            input_data = {
                'Provider': request.form['Provider'],
                'InscClaimAmtReimbursed': float(request.form['InscClaimAmtReimbursed']),
                'AttendingPhysician': request.form['AttendingPhysician'],
                'OperatingPhysician': request.form['OperatingPhysician'],
                'OtherPhysician': request.form['OtherPhysician'],
                'DeductibleAmtPaid': float(request.form['DeductibleAmtPaid']),
                'Hospitalization_Duration': int(request.form['Hospitalization_Duration']),
                'Inpatient_or_Outpatient': request.form['Inpatient_or_Outpatient'],
                'Gender': request.form['Gender'],
                'Race': request.form['Race'],
                'RenalDiseaseIndicator': request.form['RenalDiseaseIndicator'],
                'State': request.form['State'],
                'County': request.form['County'],
                'OPAnnualReimbursementAmt': float(request.form['OPAnnualReimbursementAmt']),
                'OPAnnualDeductibleAmt': float(request.form['OPAnnualDeductibleAmt']),
                'Patient_Age': int(request.form['Patient_Age']),
                'isDead': request.form['isDead'],
                'TotalClaimAmount': float(request.form['TotalClaimAmount']),
                'TotalReimbursement': float(request.form['TotalReimbursement']),
                'TotalDeductible': float(request.form['TotalDeductible'])
            }

            prediction, probability = predict_fraud_model(input_data)

            template_vars = {
                'prediction': prediction,
                'probability': probability,
                **dropdown_options  # Merge dropdown options
            }
            return render_template('predict.html', **template_vars)

        except Exception as e:
            flash(f"Error processing your request: {str(e)}", "error")
            return render_template('predict.html', **dropdown_options)

    return render_template('predict.html', **dropdown_options)


@app.route('/reporting')
def reporting():
    reporting_data = get_reporting_data(df)
    return render_template('reporting.html', **reporting_data)


@app.route('/explore')
def explore():
    return render_template('explore.html')


ALLOWED_EXTENSIONS = {'xlsx', 'xls'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/batch_analysis', methods=['GET', 'POST'])
def bulk_upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            df = pd.read_excel(file)
            output = model_bulk_predict(df)
            return send_file(
                output,
                download_name="fraud_predictions.xlsx",
                as_attachment=True,
                mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            return "Invalid file type. Only .xlsx/.xls supported", 400
    return render_template('bulk_upload.html')


@app.route('/model_evaluation')
def evaluate_model():
    metrics = get_model_metrics()
    roc_plot = plot_roc_curve()
    pr_plot = plot_precision_recall_curve()
    cm_plot = plot_confusion_matrix()
    feature_importance = get_feature_importance()
    prob_plot = plot_probability_distribution()
    shap_plot = plot_shap_summary()
    report = get_classification_report()

    return render_template('model_evaluation.html',
                           metrics=metrics,
                           roc_plot=roc_plot,
                           pr_plot=pr_plot,
                           cm_plot=cm_plot,
                           feature_importance=feature_importance,
                           prob_plot=prob_plot,
                           shap_plot=shap_plot,
                           report=report)


@app.route('/report')
def report():
    return render_template('report.html')


@app.route('/metrics')
def metrics():
    return render_template('report.html')


@app.route('/upload')
def upload():
    return render_template('upload.html')


@app.route('/bulk_upload')
def bulk_upload_error_resolve():
    return render_template('report.html')


if __name__ == '__main__':
    app.run(debug=True)
