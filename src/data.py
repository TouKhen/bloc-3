from pathlib import Path
import pandas as pd

def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Loads a CSV file from the specified path and returns its contents as a DataFrame.

    This function uses `pandas.read_csv` to read the content of the CSV file
    and parse it into a pandas DataFrame.

    :param path: The file path to the CSV file to be loaded.
    :type path: Str | Path
    :return: A pandas DataFrame containing the data from the CSV file.
    :rtype: Pd.DataFrame
    """

    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Remove customers' IDs because it's going to interfere with get_dummies
    df_no_ids = out.iloc[:,1:]

    # Convert Churn data to binary
    df_no_ids['Churn'] = df_no_ids['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    cat_cols = df_no_ids.select_dtypes(include=['object', 'category', 'bool']).columns

    # Convert TotalCharges to numeric
    df_no_ids.TotalCharges = pd.to_numeric(df_no_ids.TotalCharges, errors='coerce')
    df_no_ids.isnull().sum()

    return out
