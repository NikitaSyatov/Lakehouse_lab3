import polars as pl
from deltalake import write_deltalake
import os
from config import BRONZE_PATH, CSV_PATH, CSV_SEP, CSV_ENCODING, FL_DATE_COL, MERGE_KEYS
from lakehouse_util import merge_delta

def get_dates_from_csv() -> list:
    """Возвращает отсортированный список уникальных дат из CSV с помощью ленивого сканирования."""
    lf = pl.scan_csv(CSV_PATH, separator=CSV_SEP, encoding=CSV_ENCODING)
    dates = lf.select(pl.col(FL_DATE_COL).unique()).collect()[FL_DATE_COL].to_list()
    return sorted(dates)

def load_bronze():
    print("== Bronze: download csv per days ==")
    dates = get_dates_from_csv()
    print(f"Dates: {dates}")

    for date in dates:
        print(f"Processing date - {date}")
        # Ленивый скан с фильтром по году
        lf = (pl.scan_csv(CSV_PATH, separator=CSV_SEP, encoding=CSV_ENCODING)
              .filter(pl.col(FL_DATE_COL) == date))
        # Выполняем и получаем DataFrame
        df_day = lf.collect()
        if df_day.is_empty():
            continue

        # Используем MERGE (первый год будет overwrite, остальные append с merge)
        merge_delta(df_day, BRONZE_PATH, merge_keys=MERGE_KEYS, partition_by=None)
        print(f"  Downloaded {len(df_day)} rows from {date}")

    # После загрузки выполняем OPTIMIZE + Z-ORDER для улучшения производительности
    from lakehouse_util import optimize_delta
    optimize_delta(BRONZE_PATH, zorder_cols=["YEAR", "MONTH"])

if __name__ == "__main__":
    load_bronze()