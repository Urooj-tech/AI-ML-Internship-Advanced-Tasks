import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
import joblib

# --------- CONFIG ---------
INPUT_CSV = "telco_churn.csv"            # Your dataset
RANDOM_STATE = 42
TEST_SIZE = 0.2
OUT_JOBLIB = "churn_pipeline.joblib"
# --------------------------

def load_and_clean(path: str):
    df = pd.read_csv(path)
    if 'Churn' not in df.columns:
        raise ValueError("Expected a 'Churn' column in the dataset (target).")
    if 'TotalCharges' in df.columns and df['TotalCharges'].dtype == object:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].str.strip().replace('', np.nan), errors='coerce')
    return df

def build_preprocessor(df: pd.DataFrame):
    X = df.drop(columns=['Churn'])
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))

    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ], remainder='drop')

    return preprocessor

def main():
    data_path = Path(INPUT_CSV)
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {INPUT_CSV}")
    df = load_and_clean(str(data_path))

    X = df.drop(columns=['Churn'])
    y = df['Churn'].copy()
    if y.dtype == object or y.dtype.name == 'category':
        y = y.str.strip().map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(df)

    base_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    param_grid = [
        {
            'clf': [LogisticRegression(max_iter=2000, solver='lbfgs')],
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__penalty': ['l2'],
            'clf__class_weight': [None, 'balanced']
        },
        {
            'clf': [RandomForestClassifier(random_state=RANDOM_STATE)],
            'clf__n_estimators': [100, 200],
            'clf__max_depth': [None, 10, 20],
            'clf__class_weight': [None, 'balanced']
        }
    ]

    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=2,
        refit=True
    )

    print("Starting GridSearchCV...")
    grid.fit(X_train, y_train)

    print("\nBest params:", grid.best_params_)
    print("Best CV score (roc_auc):", grid.best_score_)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_proba))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best_model, OUT_JOBLIB)
    print(f"\nExported best pipeline to: {OUT_JOBLIB}")

if __name__ == "__main__":
    main()
