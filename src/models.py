import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Models:
    def __init__(self, df, df_dummies) -> None:
        """
        Initializes a class instance for managing and processing data using provided
        dataframes for operations. The class is designed to handle input datasets and
        their dummified versions, which are later used for machine learning.

        :param df: DataFrame containing the main dataset.
        :param df_dummies: DataFrame containing the dummified version of the dataset.

        """
        self.df = df
        self.df_dummies = df_dummies
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None


    def logistic_regression(self) -> linear_model.LogisticRegression:
        """
        Fits and trains a logistic regression model on the provided dataset.

        This method performs logistic regression on the dataset preprocessed in
        the class. It separates the feature variables from the target variable,
        splits the data into training and testing subsets, and fits the logistic
        regression model on the training data. This method finally returns the
        trained logistic regression model instance.

        :raises ValueError: If there are issues in training the model or
            with data inconsistency during operations.

        :return: A trained logistic regression model instance.
        :rtype: sklearn.linear_model._base.LogisticRegression
        """
        # Drop Churn row for the logistic regression.
        X_clean = self.df_dummies.drop(columns='Churn')
        y_clean = self.df['Churn'][X_clean.index]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_clean, y_clean,
                                                            test_size=0.3,
                                                            random_state=42)

        logreg = linear_model.LogisticRegression(max_iter=10000)
        logreg.fit(self.X_train, self.y_train)

        return logreg


    def predict_lr(self, logreg) -> pd.DataFrame:
        """
        Predicts labels for test data using a given logistic regression model.

        This method uses the provided logistic regression model to predict the
        labels for the test dataset (`X_test`). It assumes that the model has
        been trained before calling this method.

        :param logreg: The logistic regression model to use for prediction.
        :type logreg: LogisticRegression
        :return: The predicted labels for the test data.
        :rtype: numpy.ndarray
        """
        return logreg.predict(self.X_test)


    def get_feature_weights(self, logreg) -> pd.DataFrame:
        # Get feature weights from a logistic regression model
        feature_weights = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Weight': logreg.coef_[0]
        })

        return feature_weights.sort_values('Weight', ascending=False)