import polars as pl
from deltalake import write_deltalake
import os
from config import BRONZE_PATH, CSV_PATH, CSV_SEP, CSV_ENCODING, YEAR_COL, MERGE_KEYS
from lakehouse_util import merge_delta

def get_years_from_csv() -> list:
    """Возвращает список уникальных годов из CSV с помощью ленивого сканирования."""
    lf = pl.scan_csv(CSV_PATH, separator=CSV_SEP, encoding=CSV_ENCODING)
    years = lf.select(pl.col(YEAR_COL).unique()).collect()[YEAR_COL].to_list()
    return sorted(years)

def load_bronze():
    print("== Bronze: загрузка CSV по годам ==")
    years = get_years_from_csv()
    print(f"Найдены годы: {years}")

    for year in years:
        print(f"Обработка года {year}")
        # Ленивый скан с фильтром по году
        lf = (pl.scan_csv(CSV_PATH, separator=CSV_SEP, encoding=CSV_ENCODING)
              .filter(pl.col(YEAR_COL) == year))
        # Выполняем и получаем DataFrame
        df_year = lf.collect()
        if df_year.is_empty():
            continue

        # Используем MERGE (первый год будет overwrite, остальные append с merge)
        merge_delta(df_year, BRONZE_PATH, merge_keys=MERGE_KEYS, partition_by=None)
        print(f"  Загружено {len(df_year)} строк за {year}")

    # После загрузки выполняем OPTIMIZE + Z-ORDER для улучшения производительности
    from lakehouse_utils import optimize_delta
    optimize_delta(BRONZE_PATH, zorder_cols=["YEAR", "MONTH"])

if __name__ == "__main__":
    load_bronze()