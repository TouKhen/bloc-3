import pandas as pd

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds features for a given DataFrame by one-hot encoding categorical columns.

    This function processes the input DataFrame to identify columns with data types
    indicating categorical variables (object, category, and bool). It then applies
    one-hot encoding to these columns, creating binary indicator variables for each
    category and dropping the first level for each encoded column to avoid
    multicollinearity. Missing values are not encoded as separate categories.

    :param df: Input DataFrame to process.
    :type df: pd.DataFrame
    :return: A transformed DataFrame with one-hot encoded columns.
    :rtype: pd.DataFrame
    """
    cat_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    df_dummies = pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        dummy_na=False
    )

    # Drop rows with NaN values before train/test split. Since the number is low, it's not going to infer the data.
    X_clean = df_dummies.dropna()

    return X_clean

def cross_validation(models, cv=5):
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import make_scorer, accuracy_score, precision_score, \
        recall_score, f1_score

    # Define scoring metrics
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    # Create KFold cross-validator
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Dictionary to store results
    cv_results = {}

    # Perform cross-validation for each model
    for name, model in models.items():
        cv_results[name] = {}
        for metric_name, scorer in scoring.items():
            scores = cross_val_score(model['model'], model['data'].X_clean, model['data'].y_clean, cv=kf, scoring=scorer)
            cv_results[name][metric_name] = {
                'mean': scores.mean(),
                'std': scores.std()
            }

    return cv_results