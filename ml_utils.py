# %%
from typing import Tuple
import pandas as pd
import numpy as np
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

# %%


def regression_report(
    true_values: np.ndarray, predicted_values: np.ndarray
) -> None:
    """Prints out errors for performance check

    Args:
        true_values (np.ndarray): 
        predicted_values (np.ndarray): 
    """
    # accuracy check
    print(
        "Mean Absolute Error (MAE):",
        metrics.mean_absolute_error(true_values, predicted_values),
    )
    print(
        "Mean Squared Error (MSE):",
        metrics.mean_squared_error(true_values, predicted_values),
    )
    print(
        "Root Mean Squared Error (RMSE):",
        metrics.mean_squared_error(
            true_values, predicted_values, squared=False
        ),
    )
    # I would not trust the R2 as it depends too much from the number of feat
    print("R^2:", metrics.r2_score(true_values, predicted_values))


def rf_training(traindf: pd.DataFrame, trainlabel: pd.Series):
    """ Random forest training using sklearn module

    Args:
        traindf (pd.DataFrame): training data X
        trainlabel (pd.Series): labels to predict y

    Returns:
        rf estimator instance
    """
    rf = RandomForestRegressor(
        n_estimators=100, random_state=3333, n_jobs=None
    )
    rf.fit(traindf, trainlabel)
    return rf


def compute_forest_importance(
    rf_clf,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    features: list,
    permutation: str = "on",
) -> pd.Series:
    """Calculated importance of the features of the rf

    Args:
        rf_clf (estimator object implementing ‘fit’): sklearn object used to fit
        x_test (pd.DataFrame): test predictors
        y_test (pd.Series): test label
        features (list): features used to train
        permutation (str, optional): whether to perform permutain when calculating
        feature importances or not. Prevents bias in rf feature importances
        but computationally heavy. Defaults to "on".

    Returns:
        pd.Series: importance values
    """
    if permutation.lower() == "on":
        result = permutation_importance(
            rf_clf, x_test, y_test, n_repeats=10, random_state=42, n_jobs=7
        )
        forest_importances = pd.Series(result.importances_mean, index=features)
    else:
        forest_importances = pd.Series(
            rf_clf.feature_importances_, index=features
        )
    return forest_importances


def plot_forest_importance(importance_series):

    fig, ax = plt.subplots(figsize=(12, 8))
    importance_series.plot.bar(yerr=np.std(importance_series), ax=ax)
    ax.set_title("Feature importances on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()


def OOF_Predictions(
    X: np.ndarray, y: np.ndarray, model, k=5
) -> Tuple[float, float, list]:
    """Computes Out Of Fold prediction scores
    to remove dependence from
    a fixed test/validation data

    Args:
        x (np.ndarray): complete data
        model (np.ndarray): machine learning model (regressor)
        k (int, optional): folds to use in cross validation. Defaults to 5.
    Returns:
        mean_s, std_s, scores: mean and standard deviation, list of folds scores
    """
    scores = list()
    kfold = KFold(n_splits=k, shuffle=True)
    # enumerate splits
    for train_ix, test_ix in kfold.split(X):
        # get data
        train_X, test_X = X[train_ix], X[test_ix]
        train_y, test_y = y[train_ix], y[test_ix]
        # fit model
        model.fit(train_X, train_y)
        # evaluate model
        yhat = model.predict(test_X)
        # the train error is to prevent that we are not overfitting
        yhat_train = model.predict(train_X)
        mse = metrics.mean_squared_error(test_y, yhat)
        mse_train = metrics.mean_squared_error(train_y, yhat_train)
        # store score
        scores.append(mse)
        print("Test score > ", mse)
        print("Train score > ", mse_train)
    # summarize model performance
    mean_s, std_s = np.mean(scores), np.std(scores)
    return mean_s, std_s, scores
