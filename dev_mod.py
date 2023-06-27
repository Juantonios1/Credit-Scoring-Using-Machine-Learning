# CLASSIFICATION MODEL

# Library
# Basic Function
import pandas as pd
import numpy as np

# Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)

from sklearn.model_selection import cross_validate

# Ignore Warning
import warnings

warnings.filterwarnings("ignore")


# Model Function
def clas_bm(x_train, y_train, metric, cv_n=5):
    """
    input:
        x_train: df dependent variable.
        y_train: df independent variable.
        metric: 'acc', 'pre', 'rcl', 'mcc', 'f1', 'roc_auc'.
        cv_n: number of cross validation fold.
    """
    # Model Statement
    logreg = LogisticRegression(random_state=2023)
    knn = KNeighborsClassifier()
    dt = DecisionTreeClassifier(random_state=2023)
    svm_model = svm.SVC()
    gnb = GaussianNB()
    rf = RandomForestClassifier(n_estimators=100)
    ada = AdaBoostClassifier(n_estimators=100, random_state=2023)
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=1.0, max_depth=1, random_state=2023
    )

    # List
    models = [logreg, knn, dt, svm_model, gnb, rf, ada, gb]
    scoring = {
        "acc": "accuracy",
        "pre": "precision",
        "rcl": "recall",
        "mcc": "matthews_corrcoef",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    # Result List
    fit_time = []
    score_time = []
    acc_train = []
    acc_test = []
    std_acc_test = []
    pre_train = []
    pre_test = []
    std_pre_test = []
    rcl_train = []
    rcl_test = []
    std_rcl_test = []
    f1_train = []
    f1_test = []
    std_f1_test = []
    mcc_train = []
    mcc_test = []
    std_mcc_test = []
    roc_auc_train = []
    roc_auc_test = []
    std_roc_test = []

    # Cross Val Model
    for model_type in models:
        scores = cross_validate(
            model_type,
            x_train,
            y_train,
            scoring=scoring,
            cv=cv_n,
            return_train_score=True,
        )
        # Append result
        fit_time.append(scores["fit_time"].mean())
        score_time.append(scores["score_time"].mean())
        acc_train.append(scores["train_acc"].mean())
        acc_test.append(scores["test_acc"].mean())
        std_acc_test.append(scores["test_acc"].std())
        pre_train.append(scores["train_pre"].mean())
        pre_test.append(scores["test_pre"].mean())
        std_pre_test.append(scores["test_pre"].std())
        rcl_train.append(scores["train_rcl"].mean())
        rcl_test.append(scores["test_rcl"].mean())
        std_rcl_test.append(scores["test_rcl"].std())
        f1_train.append(scores["train_f1"].mean())
        f1_test.append(scores["test_f1"].mean())
        std_f1_test.append(scores["test_f1"].std())
        mcc_train.append(scores["train_mcc"].mean())
        mcc_test.append(scores["test_mcc"].mean())
        std_mcc_test.append(scores["test_mcc"].std())
        roc_auc_train.append(scores["train_roc_auc"].mean())
        roc_auc_test.append(scores["test_roc_auc"].mean())
        std_roc_test.append(scores["test_roc_auc"].std())

    # Logic for order df_bm_res
    if metric == "accuracy" or metric == "acc":
        order = "acc_test"
    elif metric == "recall" or metric == "pre":
        order = "pre_test"
    elif metric == "precision" or metric == "rcl":
        order = "rcl_test"
    elif metric == "f1":
        order = "f1_test"
    elif metric == "mcc":
        order = "mcc_test"
    elif metric == "roc_auc":
        order = "roc_auc_test"
    else:
        None

    # Result's dataframe
    df_bm_res = (
        pd.DataFrame(
            {
                "model": [
                    "Logistic Regression",
                    "KNN",
                    "Decision Tree",
                    "SVM",
                    "Naive Bayes",
                    "Random Forest",
                    "AdaBoosting",
                    "GradientBoosting",
                ],
                "score_time": score_time,
                "acc_train": acc_train,
                "acc_test": acc_test,
                "std_acc_test": std_acc_test,
                "pre_train": pre_train,
                "pre_test": pre_test,
                "std_pre_test": std_pre_test,
                "rcl_train": rcl_train,
                "rcl_test": rcl_test,
                "std_rcl_test": std_rcl_test,
                "f1_train": f1_train,
                "f1_test": f1_test,
                "std_f1_test": std_f1_test,
                "mcc_train": mcc_train,
                "mcc_test": mcc_test,
                "std_mcc_test": std_mcc_test,
                "roc_auc_train": roc_auc_train,
                "roc_auc_test": roc_auc_test,
                "std_roc_test": std_roc_test,
            }
        )
        .set_index("model")
        .sort_values(by=order, ascending=False)
    )

    return df_bm_res


# REGRESSION MODEL

# Library
# Basic Function
import pandas as pd
import numpy as np

# Model
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from sklearn.mixture import GaussianMixture

from sklearn.model_selection import cross_validate


# Model Function
def reg_bm(x_train, y_train, metric, cv_n=5):
    """
    input:
        x_train: df dependent variable.
        y_train: df independent variable.
        metric: 'r2', 'mae', 'rmse', 'mse'.
        cv_n: number of cross validation fold.
    """

    # Models
    if metric == "mae":
        dt = tree.DecisionTreeRegressor(criterion="absolute_error", random_state=2023)
    else:
        dt = tree.DecisionTreeRegressor(random_state=2023)
    knn = KNeighborsRegressor()
    svr = SVR()
    lasso = Lasso(random_state=2023)
    ridge = Ridge(random_state=2023)
    rf = RandomForestRegressor(random_state=2023)
    gb = GradientBoostingRegressor(random_state=2023)
    ada = AdaBoostRegressor(random_state=2023)
    gm = GaussianMixture(random_state=2023)

    # List
    models = [dt, knn, svr, lasso, ridge, rf, gb, ada, gm]
    scoring = {
        "r2": "r2",
        "mse": "neg_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }

    # Result List
    fit_time = []
    score_time = []
    train_r2 = []
    test_r2 = []
    std_test_r2 = []
    train_mse = []
    test_mse = []
    std_test_mse = []
    train_mae = []
    test_mae = []
    std_test_mae = []
    train_rmse = []
    test_rmse = []
    std_test_rmse = []

    # Cross Val Model
    for model_type in models:
        scores = cross_validate(
            model_type,
            x_train,
            y_train,
            cv=cv_n,
            scoring=scoring,
            return_train_score=True,
        )
        # Append result
        fit_time.append(scores["fit_time"].mean())
        score_time.append(scores["score_time"].mean())
        train_r2.append((scores["train_r2"].mean()))
        test_r2.append(scores["test_r2"].mean())
        std_test_r2.append(scores["test_r2"].std())
        train_mse.append(abs(scores["train_mse"].mean()))
        test_mse.append(abs(scores["test_mse"].mean()))
        std_test_mse.append(scores["train_mse"].std())
        train_mae.append(abs(scores["train_mae"].mean()))
        test_mae.append(abs(scores["test_mae"].mean()))
        std_test_mae.append(scores["test_mae"].std())
        train_rmse.append(abs(scores["train_rmse"].mean()))
        test_rmse.append(abs(scores["test_rmse"].mean()))
        std_test_rmse.append(scores["test_rmse"].std())

    # Dataframe
    df_bm_res = (
        pd.DataFrame(
            {
                "model": [
                    "Decision Tree",
                    "KNN",
                    "SVR",
                    "Lasso",
                    "Ridge",
                    "Random Forest",
                    "GradientBoosting",
                    "AdaBoosting",
                    "GaussianMixture",
                ],
                "fit_time": fit_time,
                "score_time": score_time,
                "train_r2": train_r2,
                "test_r2": test_r2,
                "std_test_r2": std_test_r2,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "std_test_mse": std_test_mse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "std_test_mae": std_test_mae,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "std_test_rmse": std_test_rmse,
            }
        )
        .set_index("model")
        .sort_values(by="test_" + f"{metric}", ascending=True)
    )

    return df_bm_res
