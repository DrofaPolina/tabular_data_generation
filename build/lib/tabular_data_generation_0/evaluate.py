# logreg = LogisticRegression()
# xgbclf = XGBClassifier(use_label_encoder=False)
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, pairwise, f1_score
from scipy.stats import ttest_ind, kstest, wilcoxon
import pandas as pd
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


@ignore_warnings(category=ConvergenceWarning)
def eval_classification(data, synthetic_data, target_col, classifier=LogisticRegression(), print_=True):
    output = {}
    con_data = pd.concat([data, synthetic_data], axis=0)
    con_data = con_data.dropna()
    conx_data, cony_data = con_data.drop(columns=[target_col]), con_data[target_col]
    discrete_columns = conx_data.select_dtypes(include=['object', 'datetime64']).columns
    conx_data = pd.get_dummies(conx_data, columns=discrete_columns, drop_first=True)

    x_data = conx_data[:data.shape[0]]
    synx_data = conx_data[data.shape[0]:]
    y_data = cony_data[:data.shape[0]]
    syny_data = cony_data[data.shape[0]:]

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.3, train_size=0.7, random_state=42)

    model = classifier
    model = model.fit(x_train, y_train)
    y_pred = model.predict(x_val)

    output['Original roc_auc'] = roc_auc_score(y_pred, y_val)
    if print_:
        print(f'Original roc_auc {roc_auc_score(y_pred, y_val)}')

    synx_train, synx_val, syny_train, syny_val = train_test_split(synx_data, syny_data, test_size=0.3, train_size=0.7,
                                                                  random_state=42)
    '''
    model = model.fit(synx_train, syny_train)
    syny_pred = model.predict(synx_val)
    output['\nSynthetic roc-auc'] = f1_score(syny_pred, syny_val)
    if print_:
        print(f'\nSynthetic roc-auc {f1_score(syny_pred, syny_val)}')
    '''

    model = model.fit(x_data, y_data)
    y_pred = model.predict(synx_data)
    output['\nSynthetic train, orignal test roc_auc'] = roc_auc_score(y_pred, syny_data)
    if print_:
        print('\nSynthetic train, orignal test roc_auc', roc_auc_score(y_pred, syny_data))

    model = model.fit(synx_data, syny_data)
    y_pred = model.predict(x_data)
    output['\nSynthetic test, orignal train roc_auc'] = roc_auc_score(y_pred, y_data)
    if print_:
        print('\nSynthetic test, orignal train roc_auc', roc_auc_score(y_pred, y_data))
    return output


def max_mean_discr(X, Y):
    '''
    Parameters:
    X: pd.DataFrame
    Y: pd.DataFrame
    gamma: float

    Returns: Maximum Mean Discrepancy (MMD)
    '''

    gamma = 1.0 / X.shape[1]
    XX_rbf = pairwise.rbf_kernel(X, X, gamma)
    YY_rbf = pairwise.rbf_kernel(Y, Y, gamma)
    XY_rbf = pairwise.rbf_kernel(X, Y, gamma)
    MMD = XX_rbf.mean() + YY_rbf.mean() - 2 * XY_rbf.mean()
    return MMD


def student_T(X, Y):
    '''
    Parameters:
    X: pd.DataFrame
    Y: pd.DataFrame

    Returns: list of features and t-test resulting p-values
    '''
    ttest = ttest_ind(
        np.array(X),
        np.array(Y),
        axis=0,
        equal_var=False,
        nan_policy='propagate',
        permutations=None,
        alternative='two-sided',
        trim=0
    )
    return np.round(ttest.pvalue, 4), np.round(ttest.statistic, 2)
