import polars as pl
from deltalake import write_deltalake
import os


# Configuration
CSV_PATH = "datasets/shubhamsingh42/flight-delay-dataset-2018-2024/versions/1/flight_data_2018_2024.csv"              # путь к исходному CSV
DELTA_PATH = "delta_table"         # путь к Delta-таблице
BATCH_SIZE = 10000               # количество строк в одном батче
CSV_SEP = ","                      # разделитель CSV
CSV_HAS_HEADER = True              # есть ли заголовок
CSV_ENCODING = "utf8"              # кодировка

DATE_COL = "FlightDate"
DATE_FORMAT = "%Y-%m-%d"


def process_bronze():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not exist in: {CSV_PATH}, run, please download_dataset.py")
    
    lazy_df = pl.scan_csv(
        CSV_PATH,
        separator=CSV_SEP,
        has_header=CSV_HAS_HEADER,
        encoding=CSV_ENCODING
    ).with_columns(
        pl.col(DATE_COL).str.strptime(pl.Date, format=DATE_FORMAT).alias("date")
    )
    lazy_df = lazy_df.sort("date")

    batch_iter = lazy_df.collect_batches(chunk_size=BATCH_SIZE)

    current_date = None
    buffer = []          # список DataFrame'ов для текущей даты
    first_group = True

    for batch_df in batch_iter:
        # Внутри батча данные могут содержать несколько дат,
        # но благодаря сортировке они идут строго по порядку.
        # Разбиваем батч по датам, сохраняя порядок.
        for date, group_df in batch_df.group_by("date", maintain_order=True):
            if current_date is None:
                current_date = date
                buffer.append(group_df)
            elif date == current_date:
                buffer.append(group_df)
            else:
                # Дата сменилась – записываем накопленные данные
                full_df = pl.concat(buffer)
                mode = "overwrite" if first_group else "append"
                # Удаляем служебную колонку "date", если она не нужна в финальной таблице
                write_deltalake(DELTA_PATH, full_df.drop("date"), mode=mode)
                print(f"Date {current_date}: {len(full_df)} rows")
                first_group = False
                # Начинаем новую дату
                current_date = date
                buffer = [group_df]

    # Записываем последнюю дату
    if buffer:
        full_df = pl.concat(buffer)
        mode = "overwrite" if first_group else "append"
        write_deltalake(DELTA_PATH, full_df.drop("date"), mode=mode)
        print(f"Date {current_date}: {len(full_df)} rows")

    print("✅ Download delta_table succes")


def process_silver():
    pass


def main():
    print("===========BRONZE STAGE===========")
    df_bronze = process_bronze()
    print("===========SILVER STAGE===========")
    df_silver = process_silver()


if __name__ == "__main__":
    main()