import polars as pl
from config import SILVER_PATH, GOLD_AGG_PATH, GOLD_FEATURES_PATH, DELAY_THRESHOLD
from lakehouse_util import merge_delta, optimize_delta

def build_aggregation():
    """Аналитическая витрина: средние задержки по категориям."""
    lf = pl.scan_delta(SILVER_PATH)

    # 1. Группировка по аэропорту (ORIGIN) + OP_CARRIER + hour
    agg_airport_carrier_hour = (
        lf.group_by(["ORIGIN", "OP_CARRIER", "hour"])
          .agg([
              pl.mean("ARR_DELAY").alias("avg_arr_delay"),
              pl.len().alias("num_flights")
          ])
          .collect()
    )
    # 2. Группировка по сезону + OP_CARRIER
    agg_season_carrier = (
        lf.group_by(["season", "OP_CARRIER"])
          .agg(pl.mean("ARR_DELAY").alias("avg_arr_delay"))
          .collect()
    )
    # 3. Объединяем в одну таблицу (для простоты запишем отдельные витрины)
    # Но можно сохранить и две разные таблицы.
    merge_delta(agg_airport_carrier_hour, GOLD_AGG_PATH + "_airport_carrier_hour",
                merge_keys=["ORIGIN", "OP_CARRIER", "hour"])
    merge_delta(agg_season_carrier, GOLD_AGG_PATH + "_season_carrier",
                merge_keys=["season", "OP_CARRIER"])
    print("Аналитические витрины сохранены")

def build_feature_table():
    """Feature table для ML: все признаки, включая производные."""
    lf = pl.scan_delta(SILVER_PATH)

    # Выбираем необходимые колонки для моделирования
    feature_cols = [
        "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "hour",
        "DISTANCE", "CRS_DEP_TIME", "DEP_DELAY", "CARRIER_DELAY",
        "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
        "OP_CARRIER", "ORIGIN", "DEST", "season", "route",
        "ARR_DELAY", "is_delayed"
    ]
    existing = [c for c in feature_cols if c in lf.columns]
    lf = lf.select(existing)
    df_features = lf.collect()

    # Сохраняем с MERGE (здесь ключ может быть составной рейса, но для feature table удобнее перезаписывать? Используем MERGE по тем же ключам)
    from config import MERGE_KEYS
    merge_delta(df_features, GOLD_FEATURES_PATH, merge_keys=MERGE_KEYS)
    print(f"Feature table сохранена, {len(df_features)} строк")

def build_gold():
    print("== Gold: построение витрин ==")
    build_aggregation()
    build_feature_table()
    # Оптимизация gold-таблиц
    optimize_delta(GOLD_FEATURES_PATH, zorder_cols=["YEAR", "MONTH", "OP_CARRIER"])
    optimize_delta(GOLD_AGG_PATH + "_airport_carrier_hour")

if __name__ == "__main__":
    build_gold()