# config.py
import os

# Пути к Delta-таблицам
BRONZE_PATH = "data/bronze/delta"
SILVER_PATH = "data/silver/delta"
GOLD_AGG_PATH = "data/gold/agg_delays"
GOLD_FEATURES_PATH = "data/gold/features"

# Исходный CSV (указать правильный путь к скачанному файлу)
CSV_PATH = "data/raw/flight_data_2018_2024.csv"

# Настройки загрузки
CSV_SEP = ","
CSV_ENCODING = "utf8"
YEAR_COL = "YEAR"           # колонка с годом (если есть)
FL_DATE_COL = "FL_DATE"     # колонка с датой (формат YYYY-MM-DD)

# Ключи для MERGE (уникальный идентификатор рейса)
MERGE_KEYS = ["FL_DATE", "OP_CARRIER", "FL_NUM", "ORIGIN", "DEST"]

# Порог задержки для классификации (минуты)
DELAY_THRESHOLD = 15

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "flight_delay_prediction"

# Другие настройки
ZORDER_COLS = ["YEAR", "MONTH", "OP_CARRIER"]