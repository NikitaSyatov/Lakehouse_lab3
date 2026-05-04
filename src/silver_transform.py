import polars as pl

from config import (
    BRONZE_PATH, SILVER_PATH, MERGE_KEYS, ZORDER_COLS, DELAY_THRESHOLD,
    FL_DATE_COL, YEAR_COL, MONTH_COL, DAY_OF_MONTH_COL, DAY_OF_WEEK_COL,
    CRS_DEP_TIME_COL, DEP_TIME_COL, DEP_DELAY_COL, ARR_DELAY_COL,
    CARRIER_COL, FLIGHT_NUM_COL, ORIGIN_COL, DEST_COL, DISTANCE_COL,
    CANCELLED_COL, CARRIER_DELAY_COL, WEATHER_DELAY_COL, NAS_DELAY_COL,
    SECURITY_DELAY_COL, LATE_AIRCRAFT_DELAY_COL
)
from lakehouse_util import merge_delta, optimize_delta, vacuum_delta

def transform_silver_lazy() -> pl.LazyFrame:
    """Ленивый пайплайн: читает Bronze -> очистка -> создание признаков."""
    lf = pl.scan_delta(BRONZE_PATH)

    # 1. Удалить отменённые рейсы (предполагаем колонку CANCELLED)
    lf = lf.filter(pl.col(CANCELLED_COL) == 0)

    # 2. Удалить NULL в ключевых колонках
    required_cols = [
        FL_DATE_COL, CARRIER_COL, FLIGHT_NUM_COL, ORIGIN_COL, DEST_COL,
        DEP_DELAY_COL, ARR_DELAY_COL
    ]
    lf = lf.drop_nulls(subset=required_cols)

    # 3. Фильтрация выбросов по задержкам (от -100 до 1000 минут)
    lf = lf.filter(
        (pl.col(DEP_DELAY_COL) >= -100) & (pl.col(DEP_DELAY_COL) <= 1000) &
        (pl.col(ARR_DELAY_COL) >= -100) & (pl.col(ARR_DELAY_COL) <= 1000)
    )

    # 4. Нормализация категорий (верхний регистр, удаление пробелов)
    for col in [CARRIER_COL, ORIGIN_COL, DEST_COL]:
        lf = lf.with_columns(pl.col(col).str.to_uppercase().str.strip_chars())

    # 5. Отбор нужных колонок (список желаемых признаков)
    useful_cols = [
        FL_DATE_COL, YEAR_COL, MONTH_COL, DAY_OF_MONTH_COL, DAY_OF_WEEK_COL,
        CRS_DEP_TIME_COL, DEP_TIME_COL, DEP_DELAY_COL, ARR_DELAY_COL,
        CARRIER_COL, FLIGHT_NUM_COL, ORIGIN_COL, DEST_COL, DISTANCE_COL,
        CARRIER_DELAY_COL, WEATHER_DELAY_COL, NAS_DELAY_COL, SECURITY_DELAY_COL,
        LATE_AIRCRAFT_DELAY_COL
    ]
    existing = [c for c in useful_cols if c in lf.collect_schema().names()]
    lf = lf.select(existing)

    # 6. Производные признаки
    # hour – из планового времени вылета (формат HHMM)
    lf = lf.with_columns((pl.col(CRS_DEP_TIME_COL) // 100).cast(pl.Int8).alias("hour"))

    lf = lf.with_columns(
        pl.when(pl.col(MONTH_COL).is_between(3, 5)).then(pl.lit("spring"))
        .when(pl.col(MONTH_COL).is_between(6, 8)).then(pl.lit("summer"))
        .when(pl.col(MONTH_COL).is_between(9, 11)).then(pl.lit("fall"))
        .otherwise(pl.lit("winter")).alias("season")
    )

    lf = lf.with_columns((pl.col(ORIGIN_COL) + "-" + pl.col(DEST_COL)).alias("route"))
    lf = lf.with_columns((pl.col(ARR_DELAY_COL) > DELAY_THRESHOLD).alias("is_delayed"))
    lf = lf.with_columns(pl.col(YEAR_COL).cast(pl.Int32), pl.col(MONTH_COL).cast(pl.Int32))

    return lf

def build_silver():
    print("== Silver: очистка и трансформация ==")
    lf = transform_silver_lazy()
    df_silver = lf.collect()
    print(f"Собрано {len(df_silver)} строк после очистки")

    merge_delta(df_silver, SILVER_PATH, merge_keys=MERGE_KEYS, partition_by=[YEAR_COL, MONTH_COL])
    optimize_delta(SILVER_PATH, zorder_cols=[ZORDER_COLS[-1]])
    vacuum_delta(SILVER_PATH, retention_hours=168, dry_run=False)
    print("✅ Silver-слой готов")

if __name__ == "__main__":
    build_silver()