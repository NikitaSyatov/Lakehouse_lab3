# config.py
import os

# Пути к Delta-таблицам
BRONZE_PATH = "data/bronze/delta"
SILVER_PATH = "data/silver/delta"
GOLD_AGG_PATH = "data/gold/agg_delays"
GOLD_FEATURES_PATH = "data/gold/features"

# Исходный CSV (указать правильный путь к скачанному файлу)
CSV_PATH = "data/shubhamsingh42/flight-delay-dataset-2018-2024/versions/1/flight_data_2018_2024.csv"

# Настройки загрузки
CSV_SEP = ","
CSV_ENCODING = "utf8"
BATCH_SIZE = 1000

# Названия важных колонок датасета
FL_DATE_COL = "FlightDate"                     # YYYYMMDD
YEAR_COL = "Year"
QUARTER_COL = "Quarter"
MONTH_COL = "Month"
DAY_OF_MONTH_COL = "DayofMonth"
DAY_OF_WEEK_COL = "DayOfWeek"

CARRIER_COL = "Marketing_Airline_Network"
FLIGHT_NUM_COL = "Flight_Number_Marketing_Airline"

ORIGIN_COL = "Origin"
DEST_COL = "Dest"

CRS_DEP_TIME_COL = "CRSDepTime"
DEP_TIME_COL = "DepTime"
DEP_DELAY_COL = "DepDelay"
ARR_DELAY_COL = "ArrDelay"

CANCELLED_COL = "Cancelled"
DISTANCE_COL = "Distance"

# Задержки по причинам
CARRIER_DELAY_COL = "CarrierDelay"
WEATHER_DELAY_COL = "WeatherDelay"
NAS_DELAY_COL = "NASDelay"
SECURITY_DELAY_COL = "SecurityDelay"
LATE_AIRCRAFT_DELAY_COL = "LateAircraftDelay"

# Ключи для MERGE (уникальный идентификатор рейса)
MERGE_KEYS = ["FlightDate", "Marketing_Airline_Network", "Flight_Number_Marketing_Airline", "Origin", "Dest"]

# Порог задержки для классификации (минуты)
DELAY_THRESHOLD = 15

# MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT_NAME = "flight_delay_prediction"

# Другие настройки
ZORDER_COLS = [YEAR_COL, MONTH_COL, CARRIER_COL]