import polars as pl
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from config import (
    GOLD_FEATURES_PATH, DELAY_THRESHOLD, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    ARR_DELAY_COL, CARRIER_COL, ORIGIN_COL, DEST_COL, YEAR_COL, MONTH_COL,
    DAY_OF_MONTH_COL, DAY_OF_WEEK_COL, DISTANCE_COL, CRS_DEP_TIME_COL,
    DEP_DELAY_COL, CARRIER_DELAY_COL, WEATHER_DELAY_COL, NAS_DELAY_COL,
    SECURITY_DELAY_COL, LATE_AIRCRAFT_DELAY_COL
)
from lakehouse_util import time_travel_read
from deltalake import DeltaTable

def get_feature_table(version=None):
    """Читает gold feature table, можно указать версию (time travel)."""
    if version:
        df = time_travel_read(GOLD_FEATURES_PATH, version)
    else:
        df = pl.read_delta(GOLD_FEATURES_PATH)
    return df.to_pandas()

def prepare_features(df_pd):
    """Преобразует категориальные признаки в one-hot и разделяет X, y."""
    # Целевые переменные
    y_reg = df_pd[ARR_DELAY_COL].values
    y_clf = df_pd["is_delayed"].values

    # Удаляем колонки, которые не должны быть признаками
    drop_cols = [ARR_DELAY_COL, "is_delayed"]
    X = df_pd.drop(columns=drop_cols, errors="ignore")

    # Категориальные колонки: используем константы из config
    categorical_cols = [CARRIER_COL, ORIGIN_COL, DEST_COL, "season", "route"]
    # Оставляем только те, которые есть в X
    cat_present = [c for c in categorical_cols if c in X.columns]
    X = pd.get_dummies(X, columns=cat_present, drop_first=True)

    # Заполняем NaN (если остались) нулями
    X = X.fillna(0)

    return X, y_reg, y_clf

def train_and_log():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    # Загружаем свежую версию gold feature table
    df = get_feature_table()
    X, y_reg, y_clf = prepare_features(df)

    X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    _, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

    with mlflow.start_run() as run:
        # ---------- Регрессия ----------
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_reg.fit(X_train, y_reg_train)
        y_reg_pred = rf_reg.predict(X_test)
        rmse = mean_squared_error(y_reg_test, y_reg_pred, squared=False)
        r2 = r2_score(y_reg_test, y_reg_pred)

        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(rf_reg, "regression_model")

        # ---------- Классификация ----------
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_clf_train)
        y_clf_pred = rf_clf.predict(X_test)
        acc = accuracy_score(y_clf_test, y_clf_pred)
        f1 = f1_score(y_clf_test, y_clf_pred)
        auc = roc_auc_score(y_clf_test, rf_clf.predict_proba(X_test)[:, 1])

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(rf_clf, "classification_model")

        # Feature importance для регрессии
        importances = rf_reg.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        importance_df = importance_df.sort_values("importance", ascending=False)

        importance_path = "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)

        # Логируем версию gold-таблицы
        dt = DeltaTable(GOLD_FEATURES_PATH)
        gold_version = dt.version()
        mlflow.log_param("gold_table_version", gold_version)

        print(f"Run завершён. RMSE={rmse:.2f}, R2={r2:.2f}, AUC={auc:.2f}")
        print("Топ-5 важных признаков (регрессия):")
        print(importance_df.head())

if __name__ == "__main__":
    train_and_log()