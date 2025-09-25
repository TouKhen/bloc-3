import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, f1_score, \
    accuracy_score, roc_curve, auc, roc_auc_score


class Viz:
    def __init__(self):
        pass

    def plot_corr_with_target(self, df: pd.DataFrame, target: str) -> None:
        """
        Plots the correlation of the given target variable with all other features in the
        DataFrame. The function calculates the Pearson correlation of the target variable
        with other features, sorts them in descending order, and visualizes them using
        a horizontal bar plot.

        :param df: A pandas DataFrame containing the features and the target variable.
        :type df: pd.DataFrame
        :param target: The column name of the target variable for which correlation
            is to be visualized.
        :type target: str
        :return: Does not return a value. Saves the plot as a PNG file named
            'corr_with_<target>.png' in the '../reports/figures/' directory.
        :rtype: None
        """
        corr = df.corr()[target].drop(target, errors='ignore').sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(15, 8))
        sns.barplot(x=corr.values, y=corr.index, orient='h', palette='Set2', hue=corr.index)
        ax.set_title('Correlation with Churn')
        ax.set_xlabel('Pearson correlation')
        ax.set_ylabel('Feature')
        sns.despine()
        fig.savefig(f'../reports/figures/corr_with_{target}.png')


    def simple_comparison_bar_plot(self, df: pd.DataFrame, target: str) -> None :
        """
        Create a bar plot to visualize the percentage distribution of a target column in
        a given DataFrame. The function calculates the percentage for each unique value
        in the column and displays them as a bar chart with percentage labels.

        :param df: Input DataFrame containing the dataset.
        :type df: pd.DataFrame
        :param target: Column name in the DataFrame for which the distribution is visualized.
        :type target: str
        :return: None
        """
        # Calculate percentages
        total = len(df[target])
        percentages = (df[target].value_counts() / total) * 100

        # Create a bar plot with percentages
        ax = percentages.plot(kind='bar', title=f'{target} Distribution')

        # Add percentage labels on top of each bar
        for i, v in enumerate(percentages):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

        plt.ylabel('Percentage')


    def superposed_bar_plot(self, df: pd.DataFrame, target_1: str, target_2: str) -> None:
        """
        Generates a superposed bar plot to visualize the distribution of a specified target
        variable for two subsets of data, split based on another categorical column.

        This function creates overlaid histograms for two groups within the dataset,
        distinguished by the binary values of the categorical column (`target_1`). The
        distribution is shown for another numerical or categorical column (`target_2`),
        providing insights on how the values of the numerical/categorical column vary
        across the two subsets.

        :param df: A pandas DataFrame containing the data to be visualized.
        :param target_1: The name of the categorical column is used to split the data into
            two subsets. Should have binary values (e.g., 0 and 1).
        :param target_2: The name of the numerical or categorical column for which the
            distribution is visualized using histograms.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df[df[target_1] == 0], x=target_2, bins=30, color='blue', label=f'No {target_1}', alpha=0.5)
        sns.histplot(data=df[df[target_1] == 1], x=target_2, bins=30, color='red', label=f'{target_1}', alpha=0.5)
        plt.title(f'Distribution of Customer {target_2} by {target_1} Status')
        plt.xlabel(f'{target_2} (months)')
        plt.ylabel('Count')
        plt.legend()
        plt.show()


    def stacked_bar_plot(self, df: pd.DataFrame, target_1: str, target_2: str) -> None:
        """
        Creates a stacked bar plot visualizing the proportion of a target variable
        across subcategories of another categorical variable. The proportions are
        displayed as percentages within each bar.

        :param df: Pandas DataFrame containing the data used for the plot.
        :type df: pd.DataFrame
        :param target_1: The column name for the first categorical variable to group by.
        :type target_1: str
        :param target_2: The column name for the second categorical variable
                         representing the target whose distribution is plotted.
        :type target_2: str
        :return: None. Displays a matplotlib plot.
        :rtype: None
        """
        plt.figure(figsize=(10, 6))
        churn_by_senior = df.groupby([target_1, target_2]).size().unstack()
        total = churn_by_senior.sum(axis=1)
        percentages = (churn_by_senior.div(total, axis=0) * 100).round(1)

        churn_by_senior.plot(kind='bar', stacked=True)
        plt.title(f'Churn Distribution by {target_1}')
        plt.xlabel('Senior Citizen (0=No, 1=Yes)')
        plt.ylabel('Number of Customers')
        plt.legend(title='Churn', labels=['No', 'Yes'])

        for i in range(len(churn_by_senior.index)):
            no_churn = percentages.iloc[i, 0]
            churn = percentages.iloc[i, 1]
            plt.text(i, churn_by_senior.iloc[i].sum() / 2,
                     f'No: {no_churn}%\nYes: {churn}%',
                     ha='center', va='center')

        plt.tight_layout()
        plt.show()


    def total_charges_distribution(self, df: pd.DataFrame) -> None:
        """
        Generates and displays a histogram representing the distribution of the
        'TotalCharges' column of a given DataFrame, segmented by the 'Churn'
        status. The histogram includes percentage labels for each bin, indicating
        the percentage count of entries by churn group.

        :param df: A Pandas DataFrame containing the data to be visualized.
                   The DataFrame must include columns 'TotalCharges' and 'Churn'.
        :type df: pd.DataFrame
        :return: None
        """
        plt.figure(figsize=(12, 6))
        # Calculate the percentage for each bin
        hist_data_no = plt.hist(df[df['Churn'] == 0]['TotalCharges'],
                                bins=50, alpha=0)
        hist_data_yes = plt.hist(df[df['Churn'] == 1]['TotalCharges'],
                                 bins=50, alpha=0)
        plt.clf()  # Clear the figure

        # Plot with percentages
        sns.histplot(data=df, x='TotalCharges', hue='Churn', bins=50,
                     multiple="layer", alpha=0.5)

        # Add percentage labels
        for i in range(len(hist_data_no[0])):
            if hist_data_no[0][i] > 0:  # Only add label if there are values in the bin
                plt.text(hist_data_no[1][i], hist_data_no[0][i],
                         f'{(hist_data_no[0][i] / len(df[df["Churn"] == 0]) * 100):.1f}%',
                         ha='center', va='bottom')
            if hist_data_yes[0][i] > 0:  # Only add label if there are values in the bin
                plt.text(hist_data_yes[1][i], hist_data_yes[0][i],
                         f'{(hist_data_yes[0][i] / len(df[df["Churn"] == 1]) * 100):.1f}%',
                         ha='center', va='bottom', color='orange')

        plt.title('Distribution of Total Charges by Churn Status')
        plt.xlabel('Total Charges ($)')
        plt.ylabel('Count')
        plt.legend(title='Churn', labels=['No', 'Yes'])
        plt.show()


    def feature_weights_bar(self, feature_weights: pd.DataFrame) -> None:
        """
        Generates a bar plot for visualizing feature weights from a logistic regression model.

        This method uses seaborn to create a bar plot where the weights of the features
        are displayed along the x-axis and their corresponding feature names on the y-axis.
        The output figure provides an intuitive understanding of the importance of features
        in the logistic regression model.

        :param feature_weights: A Pandas DataFrame containing two columns, 'Weight' and
            'Feature', where 'Weight' represents the corresponding feature importance and
            'Feature' provides the feature names.

        :return: None
        """
        plt.figure(figsize=(12, 8))
        sns.barplot(data=feature_weights, x='Weight', y='Feature')
        plt.title('Logistic Regression Feature Weights')
        plt.xlabel('Weight')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()


    def confusion_matrix(self, predictions, y_test) -> None:
        """
        Generates and visualizes a confusion matrix for the provided predictions and
        true labels. Calculates and displays various performance metrics such as
        accuracy, precision, specificity, and F1 score.

        :param predictions: Array of predicted labels, typically coming from
            a machine learning model's output.
        :type predictions: array-like
        :param y_test: Array of true labels against which predictions are compared.
        :type y_test: array-like
        :return: None
        """
        # Create confusion matrix
        cm = confusion_matrix(y_test, predictions)

        # Calculate specificity
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp)

        # Calculate other metrics
        precision = precision_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)

        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

        print(f'Accuracy: {accuracy_score(y_test, predictions):.3f}')
        print(f'Precision: {precision:.3f}')
        print(f'Specificity: {specificity:.3f}')
        print(f'F1 Score: {f1:.3f}')


    def roc_curve(self, y_pred_probas=None, y_test=None, labels=None) -> None:
        """
        Plots the Receiver Operating Characteristic (ROC) Curve for given predicted probabilities and
        true labels. If a list of predicted probabilities and true labels is provided, it generates
        multiple ROC curves. The Area Under the Curve (AUC) is also computed and displayed for
        each curve.

        :param y_pred_probas: List or single array-like of predicted probabilities.
        :param y_test: List or single array-like of true labels.
        :param labels: List of strings to label each ROC curve. Defaults to ['ROC curve'] if not provided.
        :return: None
        """
        # If single curve
        if not isinstance(y_pred_probas, list):
            y_pred_probas = [y_pred_probas]
            y_test = [y_test] 
            labels = ['ROC curve']

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        
        for y_pred_proba, y_true, label in zip(y_pred_probas, y_test, labels):
            fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate') 
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()


    def coef_plot(self, lr_model, subset) -> None:
        """
        Creates a bar plot of the coefficients of features for a given linear regression model
        and a subset of features. This visualization helps in understanding the importance or
        impact of each feature in the trained linear regression model.

        :param lr_model: A trained linear regression model whose coefficients will be used
            for plotting.
        :type lr_model: sklearn.linear_model._base.LinearRegression
        :param subset: A DataFrame containing the subset of features corresponding to the
            coefficients in the linear regression model.
        :type subset: pandas.DataFrame
        :return: None
        """

        coef_df = pd.DataFrame({
            'Feature': subset.columns,
            'Coefficient': lr_model.coef_
        })

        plt.figure(figsize=(10, 6))
        sns.barplot(data=coef_df, x='Coefficient', y='Feature')
        plt.title('Feature Coefficients in Linear Regression')
        plt.xlabel('Coefficient Value')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()


    def residuals_scatter(self, y_test_sub, prediction):
        """
        Compute and display a residuals scatter plot. The residuals are calculated as
        the difference between the true values (`y_test_sub`) and the predicted
        values (`prediction`). The scatter plot visualizes these residuals against the
        predicted values, which can provide insights into the prediction model's
        performance and highlight any patterns or biases.

        :param y_test_sub: Array of true target values.
        :param prediction: Array of predicted target values.
        :return: None
        """
        residuals = y_test_sub - prediction

        plt.figure(figsize=(10, 6))
        plt.scatter(prediction, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residual Plot')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.show()


    def plot_r2_poly_degree(self, poly_reg_scores):
        """
        Plot the R² scores of polynomial regression models against their polynomial degrees.
        This function visualizes the relationship between different polynomial degrees and their
        corresponding R² scores using a bar chart. Each bar represents the R² score of a specific
        polynomial degree, and its exact value is displayed as text above the bar.

        :param poly_reg_scores: A dictionary where keys represent the degree of the polynomial
            (int) and values are the corresponding R² scores (float).
        :return: None. The function displays the plot but does not return any value.
        """
        plt.figure(figsize=(8, 6))
        plt.bar(poly_reg_scores.keys(), poly_reg_scores.values())
        plt.title('R² Score by Polynomial Degree')
        plt.xlabel('Polynomial Degree')
        plt.ylabel('R² Score')
        plt.ylim(0, 1)

        for i, v in poly_reg_scores.items():
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.show()


    def plot_tree(self, clf_tree):
        """
        Visualizes a decision tree classifier using matplotlib.

        This method takes a decision tree classifier and generates a visual
        representation of the tree structure using matplotlib. The tree is
        plotted with a filled node style.

        :param clf_tree: The decision tree classifier model to be plotted.
        :type clf_tree: sklearn.tree.DecisionTreeClassifier
        :return: None
        """
        from sklearn import tree

        plt.figure(figsize=(25,20))
        tree.plot_tree(clf_tree, filled=True)


    def importance_viz_barplot(self, importances, name):
        """
        Generates a bar plot visualization of feature importances for a specific model.

        This function creates a horizontal bar plot for representing feature importance
        data by using Matplotlib and Seaborn libraries. The plot is titled to
        indicate the model's name, and it highlights the most important features
        in descending order of their importance.

        :param importances:
            A pandas DataFrame containing feature importance information. It is
            expected to have at least two columns:
            - 'feature': Names of the features.
            - 'importance': Importance scores for the features (assumed to be numerical).
        :param name:
            A string representing the name of the model whose feature
            importance is visualized.
        :return:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importances, x='importance', y='feature')
        plt.title(f'Top {len(importances)} Most Important Features in {name} Model')
        plt.xlabel('Feature Importance')
        plt.show()


    def scores_comparison_plot(self, metrics):
        """
        Generates a series of bar plots to compare a set of classification metrics across
        different models or configurations.

        The function takes in a dictionary containing metric scores (Precision, Recall,
        F1, and AUC) for each model. It creates a 2x2 grid of subplots, where each subplot
        represents one of the metrics, and displays a side-by-side bar plot for the models.
        The function ensures the y-axis for each plot ranges from 0 to 1 to enhance
        comparability.

        :param metrics: A dictionary where keys are model names or configurations and
            values are nested dictionaries containing the metrics ('Precision', 'Recall',
            'F1', and 'AUC') as keys and their corresponding scores as values.
            Example:
                {
                    "Model_1": {"Precision": 0.85, "Recall": 0.80, "F1": 0.82, "AUC": 0.90},
                    "Model_2": {"Precision": 0.83, "Recall": 0.78, "F1": 0.80, "AUC": 0.89}
                }

        :return: None
        """
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        metrics_df = pd.DataFrame(metrics).T

        # Precision plot
        metrics_df['Precision'].plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title('Precision Comparison')
        ax1.set_ylim(0, 1)

        # Recall plot
        metrics_df['Recall'].plot(kind='bar', ax=ax2, color='lightgreen')
        ax2.set_title('Recall Comparison')
        ax2.set_ylim(0, 1)

        # F1 plot
        metrics_df['F1'].plot(kind='bar', ax=ax3, color='lightcoral')
        ax3.set_title('F1 Score Comparison')
        ax3.set_ylim(0, 1)

        # AUC plot
        metrics_df['AUC'].plot(kind='bar', ax=ax4, color='plum')
        ax4.set_title('AUC Score Comparison')
        ax4.set_ylim(0, 1)

        plt.tight_layout()
        plt.show()


    def cross_validation_comparison_plot(self, models):
        """
        Generates a bar plot comparing the mean and standard deviation of cross-validation
        scores for multiple models. The input `models` should be a dictionary where keys
        are model names and values are lists of cross-validation scores.

        The function computes the mean and standard deviation for each model, and visualizes
        them using a bar plot with error bars. Each bar represents a model's average score
        with corresponding error bars indicating its standard deviation.

        :param models: A dictionary where keys are model names (str) and values are lists
            of cross-validation scores (list of floats).
        :type models: dict
        :return: None
        """
        import numpy as np

        means = {name: np.mean(scores) for name, scores in models.items()}
        stds = {name: np.std(scores) for name, scores in models.items()}

        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(means)), means.values(), yerr=stds.values(),
                capsize=5, color=['skyblue', 'lightgreen', 'salmon', 'purple', 'orange'])
        plt.xticks(range(len(means)), means.keys(), rotation=45)
        plt.title('Cross-Validation Scores Comparison')
        plt.ylabel('Mean CV Score')
        plt.ylim(0, 1)

        # Add value labels
        for i, v in enumerate(means.values()):
            plt.text(i, v + stds[list(means.keys())[i]], f'{v:.3f}',
                     ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


    def grid_search_plot(self, metrics_comparison, old_model, old_model_class, new_model, new_model_class):
        # Create subplots for each metric
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Plot each metric
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
        colors = ['skyblue', 'lightgreen']

        for metric, ax in zip(metrics, [ax1, ax2, ax3, ax4]):
            values = [metrics_comparison['Original'][metric],
                      metrics_comparison['Tuned'][metric]]
            ax.bar(['Original', 'Tuned'], values, color=colors)
            ax.set_title(f'{metric} Comparison')
            ax.set_ylim(0, 1)

            # Add value labels
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.3f}', ha='center')

        plt.tight_layout()
        plt.show()

        # Plot ROC curves for both models
        plt.figure(figsize=(8, 6))

        # Original model ROC curve
        fpr_orig, tpr_orig, _ = roc_curve(old_model_class.y_test, old_model.predict_proba(old_model_class.X_test)[:, 1])
        plt.plot(fpr_orig, tpr_orig, color='skyblue',
                 label=f'Original (AUC = {roc_auc_score(old_model_class.y_test, old_model.predict_proba(old_model_class.X_test)[:, 1]):.3f})')

        # Tuned model ROC curve
        fpr_tuned, tpr_tuned, _ = roc_curve(new_model_class.y_test, new_model.predict_proba(new_model_class.X_test)[:, 1])
        plt.plot(fpr_tuned, tpr_tuned, color='lightgreen',
                 label=f'Tuned (AUC = {roc_auc_score(new_model_class.y_test, new_model.predict_proba(new_model_class.X_test)[:, 1]):.3f})')

        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.show()