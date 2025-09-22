import pandas as pd

'''def load_dataset():
    df = pd.read_csv('data/fraud_model_data_new.csv')  # Update with actual path
    df = df.head(100)
    return df.to_dict(orient='records'), df.columns.tolist()'''


def load_dataset():
    df = pd.read_csv('data/fraud_model_data_new.csv')
    df = df.head(100)

    # Define the desired column order
    column_order = [
        'Provider',
        'InscClaimAmtReimbursed',
        'AttendingPhysician',
        'PotentialFraud',
        'TotalClaimAmount',
        'TotalReimbursement',
        'TotalDeductible',
        'OPAnnualReimbursementAmt',
        'OPAnnualDeductibleAmt',
        'DeductibleAmtPaid',
        'Patient_Age',
        'OperatingPhysician',
        'OtherPhysician',
        'Hospitalization_Duration',
        'Inpatient_or_Outpatient',
        'RenalDiseaseIndicator',
        'Gender',
        'Race',
        'State',
        'County',
        'isDead'
    ]

    # Reorder the columns in the dataframe
    df = df[column_order]

    return df.to_dict(orient='records'), column_order
