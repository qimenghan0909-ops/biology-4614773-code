from Regression.ClassicRgs import Pls, Svregression, ELM, MLR


def QuantitativeAnalysis(model, X_train, X_test, y_train, y_test, num=30, C=2, gamma=1e-07, kernel='linear'):
    if model == "Pls":
        test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae = Pls(X_train, X_test, y_train, y_test, num=num)
    elif model == "SVR":
        test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae = Svregression(X_train, X_test, y_train, y_test, C=C, gamma=gamma, kernel=kernel)
    elif model == "ELM":
        test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae = ELM(X_train, X_test, y_train, y_test)
    elif model == "MLR":
        test_rmse, test_r2, test_mae, y_pred, y_train_pred, train_rmse, train_r2, train_mae = MLR(X_train, X_test, y_train, y_test)

    return test_rmse, test_r2, test_mae, y_pred, y_test, y_train_pred, y_train, train_rmse, train_r2, train_mae



