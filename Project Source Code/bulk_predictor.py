import pandas as pd
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
from model import lgb_model, feature_columns

# Define high cardinality columns
high_card_cols = ['Provider', 'AttendingPhysician', 'OperatingPhysician', 'OtherPhysician', 'State', 'County']


def model_bulk_predict(df):
    input_df = df.copy()

    # Label encode high cardinality columns
    for col in high_card_cols:
        if col in input_df.columns:
            le = LabelEncoder()
            input_df[col] = le.fit_transform(input_df[col].astype(str))

    # One-hot encode remaining categorical variables
    low_card_cols = [col for col in input_df.select_dtypes(include=['object']).columns if col not in high_card_cols]
    input_df = pd.get_dummies(input_df, columns=low_card_cols)

    # Align columns to model's expected input
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Make predictions
    predictions = lgb_model.predict(input_df)
    probabilities = lgb_model.predict_proba(input_df)[:, 1]

    # Append results
    df['Prediction'] = predictions
    df['Probability'] = probabilities

    # Save to BytesIO for download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    output.seek(0)

    return output
