import polars as pl
from deltalake import write_deltalake
import os

from config import (
    CSV_PATH, BRONZE_PATH, CSV_SEP, CSV_ENCODING,
    FL_DATE_COL, MERGE_KEYS, ZORDER_COLS, BATCH_SIZE
)
from lakehouse_util import merge_delta, optimize_delta

def load_bronze_streaming():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)

    print("== Bronze: потоковая загрузка CSV по дням ==")
    # Ленивый скан с сортировкой по дате
    lf = pl.scan_csv(
        CSV_PATH,
        separator=CSV_SEP,
        encoding=CSV_ENCODING,
        has_header=True,
        infer_schema_length=10000
    ).sort(FL_DATE_COL)

    batch_iter = lf.collect_batches(chunk_size=BATCH_SIZE)  # батч по 10k строк

    current_date = None
    buffer = []
    first_batch = True

    for batch_df in batch_iter:
        # Разбиваем батч по датам (строки уже отсортированы)
        for date, group in batch_df.group_by(FL_DATE_COL, maintain_order=True):
            if current_date is None:
                current_date = date
                buffer.append(group)
            elif date == current_date:
                buffer.append(group)
            else:
                # Записываем накопленные строки предыдущей даты
                full_df = pl.concat(buffer)
                # mode = "overwrite" if first_batch else "append"
                # write_deltalake(BRONZE_PATH, full_df, mode=mode)
                merge_delta(full_df, BRONZE_PATH, merge_keys=MERGE_KEYS, partition_by=None)
                print(f"  Дата {current_date}: записано {len(full_df)}")
                first_batch = False
                total_dates += 1
                # Начинаем новую дату
                current_date = date
                buffer = [group]
    # Последняя дата
    if buffer:
        full_df = pl.concat(buffer)
        # mode = "overwrite" if first_batch else "append"
        # write_deltalake(BRONZE_PATH, full_df, mode=mode)
        merge_delta(full_df, BRONZE_PATH, merge_keys=MERGE_KEYS, partition_by=None)
        print(f"  Дата {current_date}: записано {len(full_df)} строк")

    # Оптимизация Delta-таблицы
    optimize_delta(BRONZE_PATH, zorder_cols=ZORDER_COLS)
    print("✅ Bronze-слой готов")

if __name__ == "__main__":
    load_bronze_streaming()