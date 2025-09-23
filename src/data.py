from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Loads a CSV file from the specified path and returns its contents as a DataFrame.

    This function uses `pandas.read_csv` to read the content of the CSV file
    and parse it into a panda DataFrame.

    :param path: The file path to the CSV file to be loaded.
    :type path: Str | Path
    :return: A panda DataFrame containing the data from the CSV file.
    :rtype: Pd.DataFrame
    """
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to load CSV file from {path}: {str(e)}")


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess a given DataFrame for further analysis.

    This function performs the following operations:
    1. Removes the first column of the DataFrame as it typically contains customer IDs,
       which may interfere with further data processing.
    2. Converts the 'Churn' column into binary values where 'Yes' is mapped to 1 and
       'No' is mapped to 0.
    3. Converts the 'TotalCharges' column to a numeric data type. Any values that cannot
       be converted to numeric are coerced to NaN.

    :param df: A panda DataFrame containing the input data to be cleaned and preprocessed.
    :type df: pd.DataFrame
    :return: The cleaned and preprocessed DataFrame with customer IDs removed, Churn
             encoded as binary, and TotalCharges converted to numeric.
    :rtype: pd.DataFrame
    """
    # Remove customers' IDs (first column) because it's going to interfere with get_dummies
    df_no_ids = df.drop(df.columns[0], axis=1)

    # Convert Churn data to binary
    df_no_ids['Churn'] = df_no_ids['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Convert TotalCharges to numeric
    df_no_ids.TotalCharges = pd.to_numeric(df_no_ids.TotalCharges, errors='coerce')
    df_no_ids.isnull().sum()

    return df_no_ids
