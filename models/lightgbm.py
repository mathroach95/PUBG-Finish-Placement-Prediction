from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from lightgbm.sklearn import LGBMRegressor


def valid_lightgbm_model(X, y, params=False):
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0xC0FFEE)
    if not params:
        model = LGBMRegressor()
    else:
        model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)

    print("Train Score : %.4f" % mean_absolute_error(y_train, pred_train))
    print("Valid Score : %.4f" % mean_absolute_error(y_valid, pred_valid))


def train_lightgbm_model(X, y, params=False):
    if not params:
        model = LGBMRegressor()
    else:
        model = LGBMRegressor(**params)
    model.fit(X, y)
    return model