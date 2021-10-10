import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error, r2_score


def my_loss_func(y_true, y_pred):
    value = np.sqrt(mean_squared_error(y_true, y_pred))
    return value


def mlr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {}
    lr = GridSearchCV(LinearRegression(), tuned_parameters, cv=kf, scoring=score)

    lr.fit(train_x, train_y)
    return lr


def ridge_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'alpha': [1, 5, 10]}
    ridge = GridSearchCV(Ridge(), tuned_parameters, cv=kf, scoring=score)

    ridge.fit(train_x, train_y)
    # print(ridge.best_params_)
    return ridge


def lasso_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'alpha': [0.01, 0.001]}
    lasso = GridSearchCV(Lasso(max_iter=100000, tol=0.01), tuned_parameters, cv=kf, scoring=score)

    lasso.fit(train_x, train_y)
    # print(lasso.best_params_)
    return lasso


def svr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'gamma': [1e-04, 1e-03], 'C': [50, 100]}
    svr = GridSearchCV(SVR(kernel='rbf'), tuned_parameters, cv=kf, scoring=score)

    svr.fit(train_x, train_y)
    # print(svr.best_params_)
    return svr


def knn_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'n_neighbors': [2, 3, 4]}
    knn = GridSearchCV(KNeighborsRegressor(), tuned_parameters, cv=kf, scoring=score)

    knn.fit(train_x, train_y)
    # print(knn.best_params_)
    return knn


def gpr_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = [{"kernel": [RBF(l) for l in [0.1]]}]
    gpr = GridSearchCV(GaussianProcessRegressor(alpha=0.001), tuned_parameters, cv=kf, scoring=score)

    gpr.fit(train_x, train_y)
    # print(gpr.best_params_)
    return gpr


def rf_predictor(train_x, train_y):
    score = make_scorer(my_loss_func, greater_is_better=False)
    kf = KFold(n_splits=10, shuffle=True, random_state=10)
    tuned_parameters = {'n_estimators': [100]}
    # tuned_parameters = {'n_estimators': [10]}
    rf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=kf, scoring=score)

    rf.fit(train_x, train_y)
    # print(rf.best_params_)
    return rf


def dt_predictor(train_x, train_y):
    tuned_parameters = {'max_depth': range(4, 10)}
    dt = GridSearchCV(DecisionTreeRegressor(), tuned_parameters, cv=10)

    # dt = DecisionTreeRegressor(max_depth=7, random_state=10)  # DK_NCOR

    dt.fit(train_x, train_y)
    # print(dt.best_params_)
    return dt


def predictors(train_x, train_y, type='MLR'):
    if type == 'SVR':
        model = svr_predictor(train_x, train_y)
    elif type == 'RF':
        model = rf_predictor(train_x, train_y)
    elif type == 'LASSO':
        model = lasso_predictor(train_x, train_y)
    elif type == 'Ridge':
        model = ridge_predictor(train_x, train_y)
    elif type == 'KNN':
        model = knn_predictor(train_x, train_y)
    elif type == 'GPR':
        model = gpr_predictor(train_x, train_y)
    elif type == 'DT':
        model = dt_predictor(train_x, train_y)
    else:
        model = mlr_predictor(train_x, train_y)
    return model
