from sklearn.preprocessing import StandardScaler
from models.lightgbm import train_lightgbm_model, valid_lightgbm_model
from utils.feature_process import feature_engineering, reduce_features_lgbm, scaling_data
from utils.data_loader import load_train_data, load_test_data, load_submission_data
from utils.hyperparameter_tunning import get_optimized_parameter



def main():
    # 테스트 데이터 로드
    train = load_train_data("train_V2.csv")
    target = "winPlacePerc"
    print("Success load data")

    # 피쳐 엔지니어링
    X = train.drop(columns=[target])
    y = train[target]
    
    X = feature_engineering(X)
    X = scaling_data(StandardScaler(), X)
    X = reduce_features_lgbm(X, y, n=20, load=True)
    columns_to_fit = X.columns
    print("Success feature engineering")

    # 학습 및 평가
    valid_lightgbm_model(X, y)
    print("Done")

    # hyperparameter_tunning & training
    params = get_optimized_parameter(X, y, K=5)
    model = train_lightgbm_model(X, y, params=params)

    # 테스트셋 학습 및 제출
    ## 테스트셋 로드 및 전처리
    test = load_test_data("test_V2.csv")
    test = feature_engineering(test)
    test = scaling_data(StandardScaler(), test)
    test = test[columns_to_fit]

    prediction = model.predict(test)

    submission = load_submission_data("sample_submission.csv")
    submission["y"] = prediction
    submission.reset_index(drop=True).to_csv(f"lgbm_submission.csv", index=False)



if __name__ == '__main__':
    main()