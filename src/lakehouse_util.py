import os
import polars as pl
from deltalake import DeltaTable, write_deltalake
import pyarrow as pa

def merge_delta(df: pl.DataFrame, path: str, merge_keys: list, partition_by: list = None):
    """
    Выполняет MERGE данных в Delta-таблицу.
    Если таблица не существует – создаёт её (с партиционированием, если указано).
    """
    if not os.path.exists(path):
        # Первая запись
        write_deltalake(
            path,
            df.to_arrow(),
            mode="overwrite",
            partition_by=partition_by
        )
        print(f"Таблица создана: {path}")
        return

    # Существующая таблица
    delta = DeltaTable(path)
    # Строим условие соединения
    predicate = " AND ".join([f"target.{k} = source.{k}" for k in merge_keys])
    # Преобразуем Polars -> Arrow
    arrow_df = df.to_arrow()
    delta.merge(
        source=arrow_df,
        predicate=predicate,
        source_alias="source",
        target_alias="target"
    ).when_matched_update_all().when_not_matched_insert_all().execute()
    print(f"MERGE выполнен в {path}, строк: {len(df)}")

def optimize_delta(path: str, zorder_cols: list = None):
    """Выполняет compaction и Z-ORDER на Delta-таблице."""
    dt = DeltaTable(path)
    dt.optimize.compact()
    if zorder_cols:
        dt.optimize.z_order(zorder_cols)
    print(f"OPTIMIZE + Z-ORDER выполнены для {path}")

def vacuum_delta(path: str, retention_hours: int = 168, dry_run: bool = False):
    """Удаляет файлы старше retention_hours."""
    dt = DeltaTable(path)
    dt.vacuum(retention_hours=retention_hours, dry_run=dry_run)
    print(f"VACUUM (dry_run={dry_run}) выполнен для {path}")

def time_travel_read(path: str, version: int) -> pl.DataFrame:
    """Читает конкретную версию Delta-таблицы через Polars."""
    return pl.read_delta(path, version=version)

def schema_evolution_example(df_new: pl.DataFrame, path: str):
    """
    Демонстрация schema evolution: запись с новыми колонками (режим merge).
    """
    write_deltalake(
        path,
        df_new.to_arrow(),
        mode="append",
        delta_write_options={"schema_mode": "merge"}
    )
    print(f"Добавлены новые колонки: {df_new.columns}")