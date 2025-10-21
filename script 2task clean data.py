import pandas as pd
import numpy as np

# Загрузка данных
file_path = r'E:\SKILLFACTORY\SF\dataSF\finalproject\data\data 2task.xlsx'
df = pd.read_excel(file_path, sheet_name='report')

print("=== НАЧАЛО ОЧИСТКИ ДАННЫХ ===")
print(f"Исходный размер: {df.shape}")

# 1. УДАЛЕНИЕ КРИТИЧЕСКИХ ОШИБОК
print("\n1. УДАЛЕНИЕ КРИТИЧЕСКИХ ОШИБОК...")

# Удаляем компании с отрицательным капиталом
initial_count = len(df)
df_clean = df[df['Капитал и резервы, RUB'] > 0].copy()
after_capital = len(df_clean)
print(f"Удалено с отрицательным капиталом: {initial_count - after_capital}")

# Удаляем компании с отрицательной дебиторкой
df_clean = df_clean[df_clean['Дебиторская задолженность, RUB'] >= 0].copy()
after_debtor = len(df_clean)
print(f"Удалено с отрицательной дебиторкой: {after_capital - after_debtor}")

# Удаляем компании с отрицательными денежными средствами
df_clean = df_clean[df_clean['Денежные средства и денежные эквиваленты, RUB'] >= 0].copy()
after_cash = len(df_clean)
print(f"Удалено с отрицательными деньгами: {after_debtor - after_cash}")

# Удаляем компании где нет данных по заёмным средствам
df_clean = df_clean[df_clean['Заёмные средства (краткосрочные), RUB'].notna()].copy()
after_loans = len(df_clean)
print(f"Удалено без данных по займам: {after_cash - after_loans}")

# 2. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ
print("\n2. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНЫХ ПЕРЕМЕННЫХ...")

# Преобразуем ССЧ в числовой формат (середина диапазона)
ssch_mapping = {
    '0 - 5': 2.5,
    '6 - 10': 8,
    '11 - 15': 13,
    '51 - 100': 75.5,
    '101 - 150': 125.5,
    '151 - 200': 175.5,
    '201 - 250': 225.5,
    '251 - 500': 375.5,
    '501 - 1 000': 750.5
}
df_clean['ссч_числовая'] = df_clean['Среднесписочная численность работников'].map(ssch_mapping)

# Создаем dummy-переменные для региона и ОПФ
df_clean = pd.get_dummies(df_clean, columns=['Регион регистрации', 'Организационно-правовая форма'], prefix=['регион', 'опф'])

# 3. СОЗДАНИЕ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ
print("\n3. СОЗДАНИЕ ЦЕЛЕВЫХ ПЕРЕМЕННЫХ...")

# Основная целевая переменная - заёмные средства
df_clean['заемные_средства'] = df_clean['Заёмные средства (краткосрочные), RUB']

# Бинарная целевая: есть/нет долга
df_clean['есть_долг'] = (df_clean['заемные_средства'] > 0).astype(int)

# Относительный показатель закредитованности
df_clean['доля_заемных'] = df_clean['Заёмные средства (краткосрочные), RUB'] / (df_clean['Заёмные средства (краткосрочные), RUB'] + df_clean['Капитал и резервы, RUB'])

# Логарифмированные версии для регрессии
df_clean['log_заемные'] = np.log1p(df_clean['заемные_средства'])
df_clean['log_выручка'] = np.log1p(df_clean['Выручка, RUB'])
df_clean['log_капитал'] = np.log1p(df_clean['Капитал и резервы, RUB'])

# 4. ЗАПОЛНЕНИЕ ПРОПУСКОВ В КЛЮЧЕВЫХ ПРИЗНАКАХ
print("\n4. ЗАПОЛНЕНИЕ ПРОПУСКОВ...")

financial_cols = ['Выручка, RUB', 'Чистая прибыль (убыток), RUB', 'Денежные средства и денежные эквиваленты, RUB', 
                 'Кредиторская задолженность, RUB', 'ссч_числовая']

for col in financial_cols:
    if col in df_clean.columns:
        missing_before = df_clean[col].isna().sum()
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        print(f"Заполнено пропусков в {col}: {missing_before}")

# 5. ФИНАЛЬНАЯ ПРОВЕРКА
print("\n5. ФИНАЛЬНАЯ ПРОВЕРКА...")
print(f"Итоговый размер данных: {df_clean.shape}")
print(f"Сохранили: {len(df_clean)/initial_count*100:.1f}% данных")

print(f"\nРаспределение по годам:")
print(df_clean['год'].value_counts().sort_index())

print(f"\nЦелевые переменные:")
print(f"Компаний с долгом: {df_clean['есть_долг'].sum()} ({df_clean['есть_долг'].mean()*100:.1f}%)")
print(f"Средний размер займа: {df_clean['заемные_средства'].median():,.0f} RUB")

# Сохранение очищенных данных
output_path = r'E:\SKILLFACTORY\SF\dataSF\finalproject\data\data_cleaned_panel.xlsx'
df_clean.to_excel(output_path, index=False)
print(f"\n✅ ОЧИЩЕННЫЕ ДАННЫЕ СОХРАНЕНЫ: {output_path}")
print(f"✅ ГОТОВО К АНАЛИЗУ! Используйте файл: data_cleaned_panel.xlsx")