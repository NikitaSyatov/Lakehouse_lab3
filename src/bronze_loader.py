import polars as pl
import time
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
        infer_schema_length=5000,
        low_memory=True
    ).sort(FL_DATE_COL)

    batch_iter = lf.collect_batches(chunk_size=BATCH_SIZE)  # батч по 10k строк

    current_date = None
    buffer = []
    total_dates = 0

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
                total_dates += 1
                # Начинаем новую дату
                current_date = date
                buffer = [group]
                time.sleep(0.1) # some pause
    # Последняя дата
    if buffer:
        full_df = pl.concat(buffer)
        # mode = "overwrite" if first_batch else "append"
        # write_deltalake(BRONZE_PATH, full_df, mode=mode)
        merge_delta(full_df, BRONZE_PATH, merge_keys=MERGE_KEYS, partition_by=None)
        print(f"  Дата {current_date}: записано {len(full_df)} строк")
        total_dates += 1

    # Оптимизация Delta-таблицы
    optimize_delta(BRONZE_PATH, zorder_cols=ZORDER_COLS)
    print(f"✅ Bronze-слой готов. Обработано дат: {total_dates}")

if __name__ == "__main__":
    load_bronze_streaming()