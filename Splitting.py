import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Printing import create_directory, create_workbook, save_to_excel_1d, save_to_excel_2d

if __name__ == '__main__':
    train_indices = []
    test_indices = []
    all_data = pd.DataFrame(pd.read_excel('./Data/data_removed_85.xlsx'))

    wb_name = './Data/data_split_85.xlsx'
    sheet_name = 'data'
    create_directory(wb_name)
    create_workbook(wb_name)

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(all_data.iloc[:, :-1], all_data.iloc[:, -1], test_size=0.2, shuffle=True)
        train_indices.append(X_train.index.values)
        test_indices.append(X_test.index.values)

    save_to_excel_2d(np.array(train_indices).transpose(), range(1, 11), wb_name, 'Train', 1, 2)
    save_to_excel_2d(np.array(test_indices).transpose(), range(1, 11), wb_name, 'Test', 1, 2)