import matplotlib
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

df_eda = pd.read_csv('data/fraud_model_data_new.csv', low_memory=False)
df_cleaned = pd.read_csv('data/cleaned_dataset.csv', low_memory=False)


def get_reporting_data(df_eda):
    output_dir = 'static/images'
    os.makedirs(output_dir, exist_ok=True)

    # System Overview
    total_unique_providers = df_eda['Provider'].nunique()
    fraud_counts = df_eda['PotentialFraud'].value_counts().to_dict()
    fraud_percentages = df_eda['PotentialFraud'].value_counts(normalize=True).to_dict()
    avg_claim = df_eda['TotalClaimAmount'].mean()
    total_claims = len(df_eda)
    highest_claim = df_eda['TotalClaimAmount'].max()

    # Flag frauds
    df_eda['FraudFlag'] = df_eda['PotentialFraud'].apply(lambda x: 1 if x == "Yes" else 0)

    # Top providers by fraud rate
    provider_stats = df_eda.groupby('Provider').agg(
        total_cases=('FraudFlag', 'count'),
        fraud_cases=('FraudFlag', 'sum'),
        avg_claim_amount=('TotalClaimAmount', 'mean')
    ).reset_index()
    provider_stats['fraud_rate'] = provider_stats['fraud_cases'] / provider_stats['total_cases']
    provider_stats_filtered = provider_stats[provider_stats['total_cases'] >= 10]
    top_providers = provider_stats_filtered.sort_values(by='fraud_rate', ascending=False).head(10).to_dict(
        orient='records')

    # Ensure claim amount is numeric
    df_eda['TotalClaimAmount'] = pd.to_numeric(df_eda['TotalClaimAmount'], errors='coerce')

    # Most fraudulent combinations of Provider + AttendingPhysician
    fraud_combo = (
        df_eda[df_eda['PotentialFraud'] == "Yes"]
        .groupby(['Provider', 'AttendingPhysician'])
        .agg(
            FraudCount=('PotentialFraud', 'count'),
            AvgClaim=('TotalReimbursement', 'mean')
        )
        .reset_index()
        .sort_values(by='FraudCount', ascending=False)
        .head(10)
    )

    top_fraud_combos = fraud_combo.to_dict(orient='records')

    # ---- Static Visualizations ----

    # 1. Chronic Conditions Proportion Bar Chart
    chronic_cols = [
        'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke', 'RenalDiseaseIndicator'
    ]

    chronic_props = df_cleaned[chronic_cols].apply(lambda col: col.value_counts(normalize=True)).T
    chronic_props.columns = ['No (0)', 'Yes (1)']

    colors = ['#4C72B0', '#DD8452']
    ax = chronic_props.plot(
        kind='bar', stacked=False, figsize=(14, 6),
        color=colors, edgecolor='black'
    )
    plt.title("Proportion of Chronic Conditions Among Patients", fontsize=16)
    plt.xlabel("Chronic Condition")
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.05)
    plt.legend(title="Condition Present")
    plt.tight_layout()
    chronic_chart_path = os.path.join(output_dir, 'chronic_conditions.png')
    plt.savefig(chronic_chart_path)
    plt.close()

    # 2. Patient Age Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_cleaned, x='Patient_Age', bins=20, kde=True, color='#4C72B0', edgecolor='black')
    plt.title('Age Distribution of Patients')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    age_dist_path = os.path.join(output_dir, 'patient_age_distribution.png')
    plt.savefig(age_dist_path)
    plt.close()

    # 3. Top 20 Attending Physicians vs Potential Fraud
    top_physicians = (
        df_cleaned[df_eda['AttendingPhysician'] != '0']['AttendingPhysician']
        .value_counts().nlargest(20).index
    )
    df_top_phys = df_eda[df_eda['AttendingPhysician'].isin(top_physicians)]
    plt.figure(figsize=(12, 6))
    sns.countplot(
        x='AttendingPhysician', hue='PotentialFraud',
        data=df_top_phys, order=top_physicians, palette='Set2'
    )
    plt.title('Top 20 Attending Physicians vs Potential Fraud')
    plt.ylabel('Number of Claims')
    plt.xlabel('Attending Physician')
    plt.xticks(rotation=45)
    plt.legend(title='Potential Fraud')
    plt.tight_layout()
    physician_fraud_path = os.path.join(output_dir, 'physician_fraud.png')
    plt.savefig(physician_fraud_path)
    plt.close()

    return {
        'total_unique_providers': total_unique_providers,
        'fraud_counts': fraud_counts,
        'fraud_percentages': fraud_percentages,
        'avg_claim': avg_claim,
        'total_claims': total_claims,
        'highest_claim': highest_claim,
        'top_providers': top_providers,
        'top_fraud_combos': top_fraud_combos,
        'chronic_chart': chronic_chart_path,
        'age_dist_chart': age_dist_path,
        'physician_fraud_chart': physician_fraud_path
    }
