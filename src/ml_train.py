import polars as pl
import os
import time
import mlflow
import mlflow.sklearn
import pandas as pd
import gc
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from config import (
    GOLD_FEATURES_PATH, DELAY_THRESHOLD, MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME,
    ARR_DELAY_COL, CARRIER_COL, ORIGIN_COL, DEST_COL, YEAR_COL, MONTH_COL,
    DAY_OF_MONTH_COL, DAY_OF_WEEK_COL, DISTANCE_COL, CRS_DEP_TIME_COL,
    DEP_DELAY_COL, CARRIER_DELAY_COL, WEATHER_DELAY_COL, NAS_DELAY_COL,
    SECURITY_DELAY_COL, LATE_AIRCRAFT_DELAY_COL
)
from lakehouse_util import time_travel_read
from deltalake import DeltaTable

def wait_for_mlflow():
    import requests
    mlflow_uri = MLFLOW_TRACKING_URI
    for i in range(30):
        try:
            response = requests.get(f"{mlflow_uri}/health", timeout=2)
            if response.status_code == 200:
                print(f"MLflow доступен по адресу {mlflow_uri}")
                return True
        except:
            pass
        print(f"Ожидание MLflow... ({i+1}/30)")
        time.sleep(2)
    print("Ошибка: MLflow не запустился")
    return False

def get_feature_table(version=None):
    """Читает gold feature table, можно указать версию (time travel)."""
    if version:
        df = time_travel_read(GOLD_FEATURES_PATH, version)
    else:
        df = pl.read_delta(GOLD_FEATURES_PATH)
    return df.to_pandas()

def prepare_features_batch(df_pd):
    """Подготовка признаков для батча с Label Encoding"""
    y_reg = df_pd["ArrDelay"].values
    y_clf = df_pd["is_delayed"].values
    
    X = df_pd.drop(columns=["ArrDelay", "is_delayed"], errors="ignore")
    
    # Label Encoding для категориальных признаков
    categorical_cols = ["Marketing_Airline_Network", "Origin", "Dest", "season", "route"]
    existing_cat = [c for c in categorical_cols if c in X.columns]
    
    for col in existing_cat:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Числовые колонки - уменьшаем тип
    for col in X.select_dtypes(include=['int64']).columns:
        X[col] = X[col].astype('int32')
    
    for col in X.select_dtypes(include=['float64']).columns:
        X[col] = X[col].astype('float32')
    
    X = X.fillna(0)
    
    return X, y_reg, y_clf

def train_incremental(max_batches=20, samples_per_batch=500):
    """
    Инкрементальное обучение на батчах
    
    Parameters:
    - max_batches: максимальное количество батчей
    - samples_per_batch: количество строк на батч
    """
    print("=== Инкрементальное обучение ===")
    
    # Читаем всю таблицу, но батчами через iter_slices
    print("Загрузка данных...")
    df_all = pl.read_delta(GOLD_FEATURES_PATH)
    total_rows = len(df_all)
    print(f"Всего строк: {total_rows}")
    
    # Разбиваем на батчи
    n_batches = min(max_batches, total_rows // samples_per_batch + 1)
    print(f"Будет обработано батчей: {n_batches}")
    
    # Инициализируем модели
    rf_reg = RandomForestRegressor(
        n_estimators=30,  # Уменьшаем для быстрого обучения
        max_depth=8,
        random_state=42,
        n_jobs=1
    )
    
    rf_clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=8,
        random_state=42,
        n_jobs=1
    )
    
    X_list = []
    y_reg_list = []
    y_clf_list = []
    
    total_processed = 0
    
    for i in range(n_batches):
        start_idx = i * samples_per_batch
        end_idx = min((i + 1) * samples_per_batch, total_rows)
        
        print(f"\nБатч {i+1}/{n_batches} (строки {start_idx}-{end_idx})")
        
        # Загружаем батч
        df_batch = df_all.slice(start_idx, samples_per_batch)
        
        if df_batch.is_empty():
            continue
        
        # Конвертируем в pandas
        df_pd = df_batch.to_pandas()
        
        # Подготовка признаков
        X_batch, y_reg_batch, y_clf_batch = prepare_features_batch(df_pd)
        
        X_list.append(X_batch)
        y_reg_list.extend(y_reg_batch)
        y_clf_list.extend(y_clf_batch)
        
        total_processed += len(X_batch)
        print(f"  Обработано строк: {len(X_batch)}, всего: {total_processed}")
        
        # Очищаем память
        del df_batch, df_pd, X_batch
        gc.collect()
    
    # Финальное обучение
    if X_list:
        print(f"\nОбучение на {len(X_list)} батчах...")
        X = pd.concat(X_list, ignore_index=True)
        y_reg = np.array(y_reg_list)
        y_clf = np.array(y_clf_list)
        
        print(f"  Размер обучающей выборки: {X.shape}")
        
        # Разделяем на train/test
        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
            X, y_reg, y_clf, test_size=0.2, random_state=42
        )
        
        # Обучение
        print("  Обучение регрессии...")
        rf_reg.fit(X_train, y_reg_train)
        
        print("  Обучение классификации...")
        rf_clf.fit(X_train, y_clf_train)
        
        # Оценка
        y_reg_pred = rf_reg.predict(X_test)
        y_clf_pred = rf_clf.predict(X_test)
        
        rmse = mean_squared_error(y_reg_test, y_reg_pred) ** 0.5
        r2 = r2_score(y_reg_test, y_reg_pred)
        acc = accuracy_score(y_clf_test, y_clf_pred)
        f1 = f1_score(y_clf_test, y_clf_pred)
        auc = roc_auc_score(y_clf_test, rf_clf.predict_proba(X_test)[:, 1])
        
        metrics = {
            "rmse": rmse,
            "r2": r2,
            "accuracy": acc,
            "f1": f1,
            "roc_auc": auc
        }
        
        print(f"\nРезультаты:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2: {r2:.2f}")
        print(f"  Accuracy: {acc:.2f}")
        print(f"  F1: {f1:.2f}")
        print(f"  ROC AUC: {auc:.2f}")
        
        return rf_reg, rf_clf, metrics
    
    return None, None, None

def train_and_log_incremental():
    """Основная функция для запуска инкрементального обучения"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    
    # Параметры обучения (настройте под свою память)
    MAX_BATCHES = 10        # Количество батчей
    SAMPLES_PER_BATCH = 500 # Строк на батч
    
    print(f"Параметры: max_batches={MAX_BATCHES}, samples_per_batch={SAMPLES_PER_BATCH}")
    
    # Инкрементальное обучение
    rf_reg, rf_clf, metrics = train_incremental(
        max_batches=MAX_BATCHES, 
        samples_per_batch=SAMPLES_PER_BATCH
    )
    
    if metrics:
        # Логируем в MLflow
        with mlflow.start_run() as run:
            mlflow.log_params({
                "max_batches": MAX_BATCHES,
                "samples_per_batch": SAMPLES_PER_BATCH,
                "total_samples": MAX_BATCHES * SAMPLES_PER_BATCH
            })
            
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            mlflow.sklearn.log_model(rf_reg, "regression_model")
            mlflow.sklearn.log_model(rf_clf, "classification_model")
            
        print(f"\n✅ Обучение завершено. Результаты сохранены в MLflow")
        print(f"   Откройте http://localhost:5000 для просмотра")
    else:
        print("❌ Обучение не выполнено")

if __name__ == "__main__":
    train_and_log_incremental()