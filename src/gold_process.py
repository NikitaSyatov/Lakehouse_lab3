# gold_build.py (или gold_process.py)
import polars as pl
from deltalake import write_deltalake   # добавить импорт
from config import (
    SILVER_PATH, GOLD_AGG_PATH, GOLD_FEATURES_PATH,
    YEAR_COL, MONTH_COL, DAY_OF_MONTH_COL, DAY_OF_WEEK_COL,
    CARRIER_COL, ORIGIN_COL, DEST_COL, DISTANCE_COL,
    CRS_DEP_TIME_COL, DEP_DELAY_COL, ARR_DELAY_COL,
    CARRIER_DELAY_COL, WEATHER_DELAY_COL, NAS_DELAY_COL,
    SECURITY_DELAY_COL, LATE_AIRCRAFT_DELAY_COL
)
from lakehouse_util import merge_delta, optimize_delta

def build_aggregation():
    """Аналитическая витрина: средние задержки по аэропорту, а/к, часу, сезону"""
    lf = pl.scan_delta(SILVER_PATH)

    # 1. По аэропорту отправления, авиакомпании и часу
    agg1 = (
        lf.group_by([ORIGIN_COL, CARRIER_COL, "hour"])
          .agg([
              pl.mean(ARR_DELAY_COL).alias("avg_arr_delay"),
              pl.len().alias("num_flights")
          ])
          .collect()
    )
    merge_delta(agg1, f"{GOLD_AGG_PATH}_airport_carrier_hour",
                merge_keys=[ORIGIN_COL, CARRIER_COL, "hour"])

    # 2. По сезону и авиакомпании
    agg2 = (
        lf.group_by(["season", CARRIER_COL])
          .agg(pl.mean(ARR_DELAY_COL).alias("avg_arr_delay"))
          .collect()
    )
    merge_delta(agg2, f"{GOLD_AGG_PATH}_season_carrier",
                merge_keys=["season", CARRIER_COL])

    print("Аналитические витрины сохранены")

def build_feature_table():
    """Feature table для ML (полная перезапись, без MERGE)"""
    lf = pl.scan_delta(SILVER_PATH)
    feature_cols = [
        YEAR_COL, MONTH_COL, DAY_OF_MONTH_COL, DAY_OF_WEEK_COL, "hour",
        DISTANCE_COL, CRS_DEP_TIME_COL, DEP_DELAY_COL, CARRIER_DELAY_COL,
        WEATHER_DELAY_COL, NAS_DELAY_COL, SECURITY_DELAY_COL, LATE_AIRCRAFT_DELAY_COL,
        CARRIER_COL, ORIGIN_COL, DEST_COL, "season", "route",
        ARR_DELAY_COL, "is_delayed"
    ]
    existing = [c for c in feature_cols if c in lf.columns]
    lf = lf.select(existing)
    df_features = lf.collect()
    # Перезаписываем таблицу целиком (mode="overwrite")
    write_deltalake(GOLD_FEATURES_PATH, df_features, mode="overwrite")
    print(f"Feature table сохранена (overwrite), {len(df_features)} строк")

def build_gold():
    print("== Gold: построение витрин ==")
    build_aggregation()
    build_feature_table()
    # Оптимизация (Z-ORDER по непартиционным колонкам)
    optimize_delta(GOLD_FEATURES_PATH, zorder_cols=[CARRIER_COL, ORIGIN_COL, DEST_COL])
    print("✅ Gold-слой готов")

if __name__ == "__main__":
    build_gold()