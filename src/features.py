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

    return pd.get_dummies(
        df,
        columns=cat_cols,
        drop_first=True,
        dummy_na=False
    )