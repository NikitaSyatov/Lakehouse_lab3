import polars as pl
from config import BRONZE_PATH, SILVER_PATH, MERGE_KEYS, ZORDER_COLS, DELAY_THRESHOLD
from lakehouse_util import merge_delta, optimize_delta, vacuum_delta

def transform_silver_lazy() -> pl.LazyFrame:
    """Ленивый пайплайн: читает Bronze -> очистка -> создание признаков."""
    lf = pl.scan_delta(BRONZE_PATH)

    # 1. Удалить отменённые рейсы (предполагаем колонку CANCELLED)
    lf = lf.filter(pl.col("CANCELLED") == 0)

    # 2. Удалить NULL в ключевых колонках
    required_cols = ["FL_DATE", "OP_CARRIER", "FL_NUM", "ORIGIN", "DEST", "DEP_DELAY", "ARR_DELAY"]
    lf = lf.drop_nulls(subset=required_cols)

    # 3. Фильтрация выбросов по задержкам (от -100 до 1000 минут)
    lf = lf.filter(
        (pl.col("DEP_DELAY") >= -100) & (pl.col("DEP_DELAY") <= 1000) &
        (pl.col("ARR_DELAY") >= -100) & (pl.col("ARR_DELAY") <= 1000)
    )

    # 4. Нормализация категорий (верхний регистр, удаление пробелов)
    for col in ["OP_CARRIER", "ORIGIN", "DEST"]:
        lf = lf.with_columns(pl.col(col).str.to_uppercase().str.strip_chars())

    # 5. Отбор нужных колонок (список желаемых признаков)
    useful_cols = [
        "FL_DATE", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
        "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY",
        "OP_CARRIER", "FL_NUM", "TAIL_NUM", "ORIGIN", "DEST", "DISTANCE",
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
    ]
    existing = [c for c in useful_cols if c in lf.columns]
    lf = lf.select(existing)

    # 6. Производные признаки
    # hour – из планового времени вылета (формат HHMM)
    if "CRS_DEP_TIME" in lf.columns:
        lf = lf.with_columns((pl.col("CRS_DEP_TIME") // 100).cast(pl.Int8).alias("hour"))

    # season – по месяцу
    if "MONTH" in lf.columns:
        lf = lf.with_columns(
            pl.when(pl.col("MONTH").is_between(3, 5)).then(pl.lit("spring"))
            .when(pl.col("MONTH").is_between(6, 8)).then(pl.lit("summer"))
            .when(pl.col("MONTH").is_between(9, 11)).then(pl.lit("fall"))
            .otherwise(pl.lit("winter")).alias("season")
        )

    # route – конкатенация ORIGIN и DEST
    if "ORIGIN" in lf.columns and "DEST" in lf.columns:
        lf = lf.with_columns((pl.col("ORIGIN") + "-" + pl.col("DEST")).alias("route"))

    # 7. Целевая колонка для классификации
    lf = lf.with_columns((pl.col("ARR_DELAY") > DELAY_THRESHOLD).alias("is_delayed"))

    # Приводим YEAR, MONTH к целым типам для партиционирования
    lf = lf.with_columns(pl.col("YEAR").cast(pl.Int32), pl.col("MONTH").cast(pl.Int32))

    return lf

def build_silver():
    print("== Silver: очистка и трансформация ==")
    # Получаем весь план трансформации (лениво)
    lf_silver = transform_silver_lazy()
    # Выполняем итеративно по датам, чтобы не перегружать память.
    # Используем collect_batches? Но для MERGE лучше одним DataFrame за раз.
    # Здесь допустим, что Silver помещается в память, иначе можно разбить по датам.
    df_silver = lf_silver.collect()
    print(f"Собрано {len(df_silver)} строк после очистки")

    # Записываем с MERGE и партиционированием по YEAR, MONTH
    merge_delta(df_silver, SILVER_PATH, merge_keys=MERGE_KEYS, partition_by=["YEAR", "MONTH"])

    # Оптимизация Silver таблицы
    optimize_delta(SILVER_PATH, zorder_cols=ZORDER_COLS)
    vacuum_delta(SILVER_PATH, retention_hours=168, dry_run=False)

if __name__ == "__main__":
    build_silver()