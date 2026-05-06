# explain_demo.py
import polars as pl
import os
from datetime import datetime
from config import SILVER_PATH, ORIGIN_COL, CARRIER_COL, ARR_DELAY_COL, YEAR_COL, MONTH_COL

def demo_explain():
    # Создаём директорию logs, если её нет
    os.makedirs("logs", exist_ok=True)
    
    # Генерируем имя файла с timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/explain_output_{timestamp}.txt"
    
    print(f"=== Демонстрация Polars explain() с pushdown оптимизациями ===")
    print(f"Результаты сохраняются в: {log_file}\n")
    
    # Открываем файл для записи
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=== Демонстрация Polars explain() с pushdown оптимизациями ===\n\n")
        
        # 1. Простой запрос с фильтрацией и агрегацией
        query1 = "1. Запрос: средняя задержка по аэропортам и авиакомпаниям для 2024 года\n"
        print(query1)
        f.write(query1 + "\n")
        
        lf = (pl.scan_delta(SILVER_PATH)
              .filter(pl.col(YEAR_COL) == 2024)  # Predicate pushdown
              .group_by([ORIGIN_COL, CARRIER_COL])
              .agg([
                  pl.mean(ARR_DELAY_COL).alias("avg_delay"),
                  pl.len().alias("flight_count")
              ])
              .sort(pl.col("avg_delay"), descending=True)
              .limit(10)
        )
        
        f.write("=== ОПТИМИЗИРОВАННЫЙ ПЛАН ЗАПРОСА 1 ===\n")
        f.write(lf.explain(optimized=True))
        f.write("\n\n" + "="*80 + "\n\n")
        
        # 2. Более сложный запрос с несколькими фильтрами
        query2 = "2. Запрос: анализ задержек по часам для рейсов с расстоянием > 500 миль\n"
        print(query2)
        f.write(query2 + "\n")
        
        lf2 = (pl.scan_delta(SILVER_PATH)
               .filter(
                   (pl.col(YEAR_COL) == 2024) &
                   (pl.col("Distance") > 500) &
                   (pl.col("DepDelay") > 0)
               )
               .group_by(["hour", CARRIER_COL])
               .agg([
                   pl.mean(ARR_DELAY_COL).alias("avg_arr_delay"),
                   pl.mean("DepDelay").alias("avg_dep_delay"),
                   pl.len().alias("cnt")
               ])
               .sort("hour")
        )
        
        f.write("=== ОПТИМИЗИРОВАННЫЙ ПЛАН ЗАПРОСА 2 ===\n")
        f.write(lf2.explain(optimized=True))
        f.write("\n\n" + "="*80 + "\n\n")
        
        # 3. Объяснение оптимизаций
        explanation = """=== ОБЪЯСНЕНИЕ PUSHDOWN ОПТИМИЗАЦИЙ ===

1. PROJECTION PUSHDOWN:
   - Polars читает только колонки, указанные в select/group_by/agg
   - В примере 1: читаются только колонки: Origin, Marketing_Airline_Network, ArrDelay, Year
   - Колонки, не участвующие в запросе (например, 'Dest', 'Route'), не загружаются

2. PREDICATE PUSHDOWN:
   - Фильтры применяются на уровне чтения данных
   - В примере 1: filter(Year == 2024) применяется до загрузки данных
   - Это уменьшает количество читаемых строк

3. AGGREGATION PUSHDOWN:
   - Агрегации (group_by) частично выполняются на уровне партиций
   - Уменьшает данные перед финальной агрегацией

4. ДОПОЛНИТЕЛЬНЫЕ ОПТИМИЗАЦИИ:
   - LIMIT pushdown: limit(10) применяется как можно раньше
   - SORT может быть частично выполнен на уровне партиций
   - Фильтры комбинируются (AND) для максимальной эффективности
"""
        f.write(explanation)
        
    print(f"\n✅ Результаты сохранены в файл: {log_file}")
    
    # Выводим краткую информацию в консоль
    print("\nКраткая информация о pushdown оптимизациях:")
    print("-" * 50)
    print("✓ PROJECTION PUSHDOWN: читаются только нужные колонки")
    print("✓ PREDICATE PUSHDOWN: фильтры применяются при чтении")
    print(f"\nПолный вывод смотрите в файле: {log_file}")

if __name__ == "__main__":
    demo_explain()