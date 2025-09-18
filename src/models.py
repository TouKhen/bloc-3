import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


class Models:
    def __init__(self, df: pd.DataFrame, df_dummies: pd.DataFrame) -> None:
        """
        Initializes a class instance for managing and processing data using provided
        dataframes for operations. The class is designed to handle input datasets and
        their dummified versions, which are later used for machine learning.

        :param df: DataFrame containing the main dataset.
        :param df_dummies: DataFrame containing the dummified version of the dataset.

        """
        self.df = df
        self.df_dummies = df_dummies


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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

        logreg = linear_model.LogisticRegression(max_iter=10000)
        logreg.fit(self.X_train, self.y_train)

        return logreg


    def linear_regression(self, X_dummies: pd.DataFrame, subset: list, y: pd.Series) -> object:
        """
        Fits a linear regression model to a subset of features and target variable. This function performs
        a train-test split on the provided subset of features and target variable, and then fits a linear
        regression model to the training data. It returns the fitted regression model.

        :param X_dummies: The complete set of dummy-encoded features.
        :type X_dummies: pandas.DataFrame
        :param subset: A list of column names indicating the subset of features to use for the model.
        :type subset: list
        :param y: The target variable for the model.
        :type y: pandas.Series
        :return: A fitted linear regression model.
        :rtype: sklearn.linear_model.LinearRegression
        """
        X_subset = X_dummies[subset]

        self.X_train_sub, self.X_test_sub, self.y_train_sub, self.y_test_sub = train_test_split(X_subset, y,
                                                                            test_size=0.3,
                                                                            random_state=42)

        lr_model = linear_model.LinearRegression()
        return lr_model.fit(self.X_train_sub, self.y_train_sub)


    def get_subset_coefs(self, model) -> pd.DataFrame:
        """
        Extracts the coefficients of the model for the subset of features and
        returns a DataFrame containing feature names and their corresponding
        coefficients. This method is useful for interpreting the model's
        behavior specifically for the selected subset of features.

        :param model: The fitted model from which to extract coefficients.
                      The model must have a "coef_" attribute available.
        :type model: Any
        :return: A DataFrame with two columns: 'Feature' and 'Coefficient',
                 where 'Feature' represents the feature names of the subset
                 of input data and 'Coefficient' represents their corresponding
                 coefficients from the model.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame({
            'Feature': self.X_subset.columns,
            'Coefficient': model.coef_
        })


    def universal_predict(self, model) -> pd.DataFrame:
        """
        Predicts outcomes using the provided model and the test dataset.

        This method applies the `predict` function of the given model to the
        test dataset (self.X_test) and returns the predictions. The model
        should be compatible with the data provided in `self.X_test`.

        :param model: The machine learning model with a `predict` method.
        :type model: Any
        :return: A DataFrame containing predictions from the model.
        :rtype: pd.DataFrame
        """
        return model.predict(self.X_test)


    def selected_predict(self, model, selected_features) -> pd.DataFrame:
        """
        Makes predictions using the provided model based on the selected features.

        This method takes a pre-trained model and a list of selected features,
        applies these features to the test data, and returns the predictions
        generated by the model.

        :param model: Trained prediction model that implements a `predict` method.
        :param selected_features: List of feature names to be used for prediction.
        :type selected_features: list
        :return: Predictions made by the model for the selected features in the test data.
        :rtype: pd.DataFrame
        """
        return model.predict(self.X_test[selected_features])


    def get_feature_weights(self, logreg) -> pd.DataFrame:
        """
        Extracts the feature weights from a logistic regression model. This method generates a
        DataFrame containing the features from the test set and their corresponding weights
        from the provided logistic regression model. The resulting DataFrame is sorted in
        descending order by the weights.

        :param logreg: A fitted logistic regression model from which feature weights are
            extracted.
        :type logreg: sklearn.linear_model.LogisticRegression
        :return: A DataFrame containing the features and their weights from the logistic
            regression model, sorted in descending order by weight.
        :rtype: pd.DataFrame
        """
        # Get feature weights from a logistic regression model
        feature_weights = pd.DataFrame({
            'Feature': self.X_test.columns,
            'Weight': logreg.coef_[0]
        })

        return feature_weights.sort_values('Weight', ascending=False)