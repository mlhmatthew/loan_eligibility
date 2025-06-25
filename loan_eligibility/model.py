from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

def train_model(df):
    # Drop columns if they exist
    X = df.drop(columns=[col for col in ['Loan_ID', 'Loan_Approved'] if col in df.columns])

    # Get feature column order
    feature_order = list(X.columns)

    if 'Loan_Approved' in df.columns:
        y = df['Loan_Approved']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        return model, X_test, y_test, feature_order
    else:
        model = LogisticRegression()
        return model, None, None, feature_order



def evaluate_model(model, X_test, y_test):
    if X_test is not None and y_test is not None:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        return acc, cm
    else:
        return None, None
