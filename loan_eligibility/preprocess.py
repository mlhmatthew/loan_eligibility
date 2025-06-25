import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess(df):
    df = df.copy()

    # Clean 'Dependents' column if it exists
    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].replace('3+', 3)
        df['Dependents'] = df['Dependents'].astype(float)

    # Fill missing values safely
    df.fillna({
        'Gender': df['Gender'].mode()[0] if 'Gender' in df else None,
        'Married': df['Married'].mode()[0] if 'Married' in df else None,
        'Self_Employed': df['Self_Employed'].mode()[0] if 'Self_Employed' in df else None,
        'Credit_History': df['Credit_History'].mode()[0] if 'Credit_History' in df else None,
        'LoanAmount': df['LoanAmount'].median() if 'LoanAmount' in df else None,
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0] if 'Loan_Amount_Term' in df else None,
    }, inplace=True)

    # Encode categorical variables only if present
    categorical = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Approved']
    for col in categorical:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Final cleanup: fill any remaining NaNs
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

    return df
