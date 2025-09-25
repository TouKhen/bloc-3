import pandas as pd
import numpy as np
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
        # Keep reference for get_subset_coefs()
        self.X_subset = X_subset

        # Align y with X_subset's index to ensure identical sizes
        y_aligned = y.reindex(X_subset.index)

        # If any NaN values remain, filter them safely
        mask = y_aligned.notna()
        if mask.any() and (~mask).any():
            X_subset = X_subset.loc[mask]
            y_aligned = y_aligned.loc[mask]

        self.X_train_sub, self.X_test_sub, self.y_train_sub, self.y_test_sub = train_test_split(
            X_subset,
            y_aligned,
            test_size=0.3,
            random_state=42
        )

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


    def selected_predict(self, model) -> pd.DataFrame:
        """
        Executes the prediction process using a given trained model on a
        subset of the test dataset. This function is designed for cases
        where only a specific subset of the test data is required for
        prediction.

        :param model: The trained model used for making predictions.
        :type model: Any
        :return: The prediction results as a DataFrame.
        :rtype: pd.DataFrame
        """
        return model.predict(self.X_test_sub)


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


    def get_r2_score(self, predictions):
        """
        Calculates the R^2 (coefficient of determination) regression score for the
        provided predictions in comparison to the true target output.

        :param predictions: The predicted values to be compared against the true
            values. Should be an iterable structure containing predictions.
        :type predictions: Iterable
        :return: The calculated R^2 score, which quantifies the model's performance.
            A value of 1.0 indicates perfect prediction, while a lower value
            represents worse predictive performance.
        :rtype: float
        """
        from sklearn.metrics import r2_score

        return r2_score(self.y_test_sub, predictions)


    def get_rmse(self, predictions):
        """
        Computes the Root Mean Squared Error (RMSE) for the given predictions using the
        true test data.

        :param predictions: The predicted values obtained from the model.
        :type predictions: array-like
        :return: The computed Root Mean Squared Error (RMSE) value.
        :rtype: float
        """
        from sklearn.metrics import mean_squared_error

        return np.sqrt(mean_squared_error(self.y_test_sub, predictions))


    def polynomial_features(self, X_poly, y_poly, degree):
        """
        Generates polynomial features for the given input data. This function splits the input
        data into training and testing sets and returns an instance of the PolynomialFeatures
        transformation.

        :param X_poly: The input feature set to be transformed.
        :param y_poly: The target (dependent variable) corresponding to the input features.
        :param degree: The degree of the polynomial features to be generated.
        :return: Instance of PolynomialFeatures initialized with the specified degree.
        """
        from sklearn.preprocessing import PolynomialFeatures

        X_subset = X_poly

        # Align y with X_subset's index to ensure identical sizes
        y_aligned = y_poly.reindex(X_subset.index)

        # If any NaN values remain, filter them safely
        mask = y_aligned.notna()
        if mask.any() and (~mask).any():
            X_subset = X_subset.loc[mask]
            y_aligned = y_aligned.loc[mask]

        # Split data
        self.X_train_poly, self.X_test_poly, self.y_train_poly, self.y_test_poly = train_test_split(X_subset, y_aligned, test_size=0.3, random_state=42)

        return PolynomialFeatures(degree)


    def find_best_degree(self, X_poly, y_poly, degrees):
        """
        Finds the best fitting polynomial degree by training models for each degree and
        evaluating their performance.

        This method iterates through a list of polynomial degrees, constructs a pipeline
        with polynomial feature transformation and linear regression for each degree,
        trains the pipeline on training data, and evaluates each model's score on the
        test data. The method returns both the fitted models and their respective scores.

        :param degrees: List of integers representing polynomial degrees to evaluate.
        :type degrees: list[int]
        :return: A tuple containing two entities:
            1. Dictionary mapping degrees to their corresponding trained pipeline models.
            2. Dictionary mapping degrees to their respective evaluation scores.
        :rtype: tuple[dict[int, sklearn.pipeline.Pipeline], dict[int, float]]
        """
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline

        models = {}
        scores = {}

        for degree in degrees:
            model = make_pipeline(
                self.polynomial_features(X_poly, y_poly, degree),
                LinearRegression()
            )

            model.fit(self.X_train_poly, self.y_train_poly)
            models[degree] = model
            scores[degree] = model.score(self.X_test_poly, self.y_test_poly)

        return models, scores


    def decision_tree(self, params=None):
        """
        Trains a decision tree classifier using the provided dataset. The function splits
        the data into training and testing sets, initializes a decision tree classifier,
        and trains it with the training set.

        :raises KeyError: If the 'Churn' column is missing in the dataset during processing.
        :raises ValueError: If the dataset contains invalid or inconsistent data.

        :return: Trained decision tree classifier
        :rtype: sklearn.tree.DecisionTreeClassifier
        """
        from sklearn import tree

        # Drop the Churn row for the decision tree.
        X_clean = self.df_dummies.drop(columns='Churn')
        y_clean = self.df['Churn'][X_clean.index]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

        clf_tree = tree.DecisionTreeClassifier(**params) if params else tree.DecisionTreeClassifier()

        return clf_tree.fit(self.X_train, self.y_train)


    def random_forest(self, params=None):
        """
        Trains a Random Forest classifier on the dataset, splitting it into training and testing
        sets, and returns the fitted model. The target variable 'Churn' is excluded from input
        features during training.

        :raises KeyError: If the 'Churn' column is not found in the dataset.
        :raises ValueError: If the dataset is not suitable for splitting or training a model.
        :return: Trained Random Forest classifier.
        :rtype: sklearn.ensemble.RandomForestClassifier
        """
        from sklearn.ensemble import RandomForestClassifier

        # Drop the Churn row for the random forest.
        X_clean = self.df_dummies.drop(columns='Churn')
        y_clean = self.df['Churn'][X_clean.index]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_clean, y_clean, test_size=0.3, random_state=42)

        clf_forest = RandomForestClassifier(random_state=42, **params) if params else RandomForestClassifier(
            random_state=42
        )

        return clf_forest.fit(self.X_train, self.y_train)


    def grid_search(self, model, parameters):
        from sklearn.model_selection import GridSearchCV

        svc_grid = GridSearchCV(model, parameters, cv=5, scoring='accuracy')
        return svc_grid.fit(self.X_train, self.y_train)