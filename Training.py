import copy

import numpy as np
import pandas as pd
from ModelSelection import predictors
from Printing import create_workbook, create_directory, save_to_excel_1d
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score


def model_selection(data, trains, tests, models, upper_path, n, ):
    for model_name in models:
        wb = upper_path + '/' + model_name + '.xlsx'
        cv_rmses, test_rmses, test_mapes, test_r2s, coefs, inters, importance = [], [], [], [], [], [], []

        create_directory(wb)
        create_workbook(wb)
        for i in range(n):
            train_x = data[trains[i], :-1]
            train_y = data[trains[i], -1]
            test_x = data[tests[i], :-1]
            test_y = data[tests[i], -1]
            # scaler = preprocessing.StandardScaler()
            # pro_train_x = scaler.fit_transform(train_x)
            # pro_test_x = scaler.transform(test_x)
            pro_train_x = copy.deepcopy(train_x)
            pro_test_x = copy.deepcopy(test_x)

            model = predictors(pro_train_x, train_y, type=model_name)
            best_model = model.best_estimator_
            avg_rmse = np.abs(model.best_score_)
            cv_rmses.append(avg_rmse)
            best_model.fit(pro_train_x, train_y)
            predict_test = best_model.predict(pro_test_x)
            test_rmse = np.sqrt(mean_squared_error(test_y, predict_test))
            test_mape = sum(abs((predict_test - test_y) / test_y ) / len(test_y))
            test_r2 = r2_score(test_y, predict_test)
            test_rmses.append(test_rmse)
            test_mapes.append(test_mape)
            test_r2s.append(test_r2)
            print('CV RMSE: %.5f;   Test RMSE: %.5f;   Test MAPE: %.5F;   Test R2: %.5f' % (avg_rmse, test_rmse, test_mape, test_r2))

            save_to_excel_1d(test_y, str(i), wb, 'true values', i+1, 2)
            save_to_excel_1d(predict_test, str(i), wb, 'predicted values', i+1, 2)

            if model_name in ['MLR', 'Ridge', 'LASSO']:
                importance = np.append(best_model.coef_, best_model.intercept_)
            if model_name in 'RF':
                importance = best_model.feature_importances_
            if len(importance):
                save_to_excel_1d(importance, str(i) + 'th', wb, 'importance', i+1, 2)

        sheet = 'performance'
        save_to_excel_1d(cv_rmses, 'CV RMSE', wb, sheet, 1, 2)
        save_to_excel_1d(test_rmses, 'Test RMSE', wb, sheet, 2, 2)
        save_to_excel_1d(test_mapes, 'Test MAPE', wb, sheet, 3, 2)
        save_to_excel_1d(test_r2s, 'Test R2', wb, sheet, 4, 2)


if __name__ == '__main__':
    wb_data = './Data/data_revised_85.xlsx'
    wb_splitting = './Data/data_split_85.xlsx'
    data = pd.read_excel(wb_data, sheet_name='data').values
    train_indices = pd.read_excel(wb_splitting, sheet_name='Train').values.T
    test_indices = pd.read_excel(wb_splitting, sheet_name='Test').values.T

    models = ['MLR', 'SVR', 'KNN', 'GPR', 'Ridge', 'LASSO', 'RF']
    path = 'Result/DataWithoutNormalization/Prediction'
    create_directory(path)
    model_selection(data, train_indices, test_indices, models, path, 10)
