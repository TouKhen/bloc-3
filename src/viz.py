import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_corr_with_target(df: pd.DataFrame, target: str) -> None:
    corr = df.corr()[target].drop(target, errors='ignore').sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(15, 8))
    sns.barplot(x=corr.values, y=corr.index, orient='h', palette='Set2', hue=corr.index)
    ax.set_title('Correlation with Churn')
    ax.set_xlabel('Pearson correlation')
    ax.set_ylabel('Feature')
    sns.despine()