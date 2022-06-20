import pandas as pd
import numpy as np
import os
from lightgbm.sklearn import LGBMRegressor
from sklearn.feature_selection import RFE
from reduce_memory import reduce_mem_usage

def mean_rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])

def add_feature(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    df = df.merge(agg, how='left', on=['groupId'])
    df["totalDistance"] = df['walkDistance'] + df['rideDistance'] + df['swimDistance']
    df['headshotKills_over_kills'] = df['headshotKills'] / (df['kills'] + 0.0001)
    df['rideDistance_over_totalDistance'] = df['rideDistance'] / (df["totalDistance"] + 0.0001) 
    df['weaponsAcquired_over_totalDistance'] = df['weaponsAcquired'] / (df["totalDistance"] + 0.0001) 
    df['kills_over_damageDealt'] = df['kills'] / (df['damageDealt']+ 0.0001) 
    agg = pd.pivot_table(data=df,index="groupId", values="revives", aggfunc="sum").rename(columns={"revives" : 'totalrevives'})
    df = df.merge(agg, how='left', on=['groupId'])
    agg = pd.pivot_table(data=df,index="groupId", values=["assists", "kills"], aggfunc="sum")
    agg["totalassists_over_totalkills"] = agg["assists"]/(agg["kills"] + 0.0001)
    df = df.merge(agg[["totalassists_over_totalkills"]], how='left', on=['groupId'])
    agg = df.groupby(['matchId']).size().to_frame('players_in_match')
    df = df.merge(agg, how='left', on=['matchId'])
    df["killnorm"] = df["kills"] / df["players_in_match"]
    df['heals_over_matchDuration'] = df['heals'].astype(np.float32) / (df['matchDuration'] + 0.001)
    df['boosts_over_matchDuration'] = df['boosts'].astype(np.float32) / (df['matchDuration'] + 0.001)
    df['walkDistance_over_matchDuration'] = df['walkDistance'].astype(np.float32) / (df['matchDuration'] + 0.001)
    df['totalDistance_over_matchDuration'] = (df["totalDistance"]) / (df['matchDuration'] + 0.001)
    return df

def feature_engineering(X):
    X.drop(columns=['matchType',"killPlace"], inplace=True)
    X = add_feature(X)
    X = mean_rank_by_team(X)
    X = reduce_mem_usage(X)
    return X

def scaling_data(scaler, X):
    trainX = X.drop(["Id", "groupId", "matchId", "matchType"], axis=1)
    trainX_col = trainX.columns
    trainX = scaler.fit_transform(trainX)
    trainX = pd.DataFrame(trainX, columns=trainX_col)
    return trainX


def drop_anormal(df):
    # 결측치 제거
    df.dropna(subset=["winPlacePerc"], inpalce=True)

    # 이상치 제거
    t = df.groupby(['matchId'])[["winPlacePerc"]].max()
    match_to_drop = t[t["winPlacePerc"] < 1].index
    df.drop(df[df["matchId"].isin(match_to_drop)].index, axis=0, inplace=True)
    df.reset_index(inplace = True, drop = True)

    return df

def reduce_features_lgbm(trainX, y, n, load=False):
    if load and os.path.isfile("checkpoints/feature_rank.csv"):
        features = pd.read_csv("checkpoints/feature_rank.csv")
        important_features = features.sort_values(by="rank").head(n).feature.values
        return trainX[important_features]

    model = LGBMRegressor()
    rfe = RFE(model, n_features_to_select=n, verbose=1)
    rfe.fit(trainX, y)

    important_features = []
    for col, isTrue in zip(rfe.feature_names_in_, rfe.support_):
        if isTrue:
            important_features.append(col)

    X = trainX[important_features]
    features = pd.DataFrame(data=list(zip(rfe.feature_names_in_, rfe.support_, rfe.ranking_)), columns=["feature", "support", "rank"])
    features.to_csv("checkpoints/feature_rank.csv", index=False)
    print(f"Features Extracted")
    return X


