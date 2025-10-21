import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

file_path = r'data\data_cleaned_panel.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

print("=== КОРРЕКТНЫЙ АНАЛИЗ ФАКТОРОВ ВЛИЯНИЯ НА ЗАКРЕДИТОВАННОСТЬ ===")

# 1. СОЗДАЕМ ПРИЗНАКИ БЕЗ ОШИБОК
print("1. ПОДГОТОВКА ДАННЫХ")

# Базовые финансовые показатели
financial_ratios = {
    'рентабельность': lambda x: x['Чистая прибыль (убыток), RUB'] / (x['Выручка, RUB'] + 1),
    'выручка_на_сотрудника': lambda x: x['Выручка, RUB'] / (x['ссч_числовая'] + 1),
    'капитал_к_выручке': lambda x: x['Капитал и резервы, RUB'] / (x['Выручка, RUB'] + 1),
    'доля_кредиторки': lambda x: x['Кредиторская задолженность, RUB'] / (x['Выручка, RUB'] + 1),
    'прибыль_на_сотрудника': lambda x: x['Чистая прибыль (убыток), RUB'] / (x['ссч_числовая'] + 1)
}

# Создаем признаки безопасно
for name, func in financial_ratios.items():
    try:
        df[name] = func(df)
        # Защита от бесконечных значений и пропусков
        df[name] = df[name].replace([np.inf, -np.inf], np.nan)
        df[name] = df[name].fillna(df[name].median())
    except Exception as e:
        print(f"Ошибка при создании признака {name}: {e}")

print(f"Создано {len(financial_ratios)} финансовых показателей")

# 2. ФОРМИРУЕМ ПРИЗНАКИ
print("\n2. ФОРМИРОВАНИЕ ПРИЗНАКОВ")

# Базовые числовые признаки (проверяем наличие)
base_numeric = [
    'log_выручка', 'log_капитал', 'Чистая прибыль (убыток), RUB',
    'Денежные средства и денежные эквиваленты, RUB', 'Кредиторская задолженность, RUB',
    'ссч_числовая', 'год'
]

# Добавляем созданные финансовые показатели
base_features = [f for f in base_numeric if f in df.columns] + list(financial_ratios.keys())

# Категориальные переменные (ограничиваем количество)
region_cols = [col for col in df.columns if col.startswith('регион_')][:15]  # топ-15 регионов
base_features.extend(region_cols)

opf_cols = [col for col in df.columns if col.startswith('опф_')]
base_features.extend(opf_cols)

# Создаем финальный набор
X = df[base_features].copy()
y = df['log_заемные'].copy()

# Заполняем пропуски медианой
X = X.fillna(X.median())

print(f"Финальный набор: {X.shape[0]} наблюдений, {X.shape[1]} признаков")

# 3. СТРОИМ МОДЕЛЬ
print("\n3. ПОСТРОЕНИЕ МОДЕЛИ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Оптимизированная модель
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

# Кросс-валидация
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f"R² на тесте: {r2:.4f}")
print(f"MSE на тесте: {mse:.4f}")
print(f"Кросс-валидация R²: {cv_scores.mean():.4f}")

# 4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
print("\n4. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")

importance_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

print("Топ-15 самых важных признаков:")
print("=" * 50)
for i, row in importance_df.head(15).iterrows():
    correlation = X[row['Признак']].corr(y)
    direction = "↑" if correlation > 0 else "↓"
    print(f"{i+1:2d}. {row['Признак']:30} : {row['Важность']:.4f} {direction}")

# 5. ГРУППИРОВКА ПРИЗНАКОВ
print("\n5. АНАЛИЗ ПО ГРУППАМ ПРИЗНАКОВ")

# Классифицируем признаки
regional_features = [f for f in importance_df['Признак'] if 'регион_' in f]
opf_features = [f for f in importance_df['Признак'] if 'опф_' in f]
financial_features = [f for f in importance_df['Признак'] if f in financial_ratios.keys()]
numeric_features = [f for f in importance_df['Признак'] if f in base_numeric and f not in financial_features]

# Суммарная важность
regional_importance = importance_df[importance_df['Признак'].isin(regional_features)]['Важность'].sum()
opf_importance = importance_df[importance_df['Признак'].isin(opf_features)]['Важность'].sum()
financial_importance = importance_df[importance_df['Признак'].isin(financial_features)]['Важность'].sum()
numeric_importance = importance_df[importance_df['Признак'].isin(numeric_features)]['Важность'].sum()

print(f"Региональные факторы: {regional_importance:.3f}")
print(f"Организационные формы: {opf_importance:.3f}")
print(f"Финансовые ratios: {financial_importance:.3f}")
print(f"Числовые показатели: {numeric_importance:.3f}")

# 6. ДЕТАЛЬНЫЙ АНАЛИЗ ТОП-ФАКТОРОВ БЕЗ ОШИБОК
print("\n6. ДЕТАЛЬНЫЙ АНАЛИЗ ТОП-ФАКТОРОВ")

top_5_features = importance_df.head(5)

for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    feature_name = row['Признак']
    feature_data = X[feature_name]
    
    print(f"\n{i}. {feature_name}:")
    print(f"   Важность: {row['Важность']:.4f}")
    print(f"   Корреляция с займами: {feature_data.corr(y):.4f}")
    
    # Безопасный анализ распределения
    try:
        # Используем квантили вместо группировки
        low_q = feature_data.quantile(0.25)
        median_val = feature_data.median()
        high_q = feature_data.quantile(0.75)
        
        print(f"   Распределение: 25%={low_q:.2f}, 50%={median_val:.2f}, 75%={high_q:.2f}")
        
    except Exception as e:
        print(f"   Ошибка при анализе распределения: {e}")

# 7. ВИЗУАЛИЗАЦИЯ БЕЗ ОШИБОК
print("\n7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# График 1: Важность признаков
top_12 = importance_df.head(12)
sns.barplot(data=top_12, x='Важность', y='Признак', ax=ax1)
ax1.set_title('Топ-12 факторов влияния')

# График 2: Распределение важности по группам
groups = ['Регионы', 'ОПФ', 'Финансы', 'Числовые']
importance_values = [regional_importance, opf_importance, financial_importance, numeric_importance]
ax2.bar(groups, importance_values, color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
ax2.set_title('Важность по группам факторов')
ax2.set_ylabel('Суммарная важность')

# График 3: Качество предсказаний
ax3.scatter(y_test, y_pred, alpha=0.6, s=30)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax3.set_xlabel('Реальные значения')
ax3.set_ylabel('Предсказания')
ax3.set_title(f'Качество модели (R² = {r2:.3f})')

# График 4: Сравнение с предыдущей моделью
models = ['Базовая', 'Текущая']
r2_scores = [0.2574, r2]
ax4.bar(models, r2_scores, color=['lightblue', 'lightgreen'])
ax4.set_ylabel('R²')
ax4.set_title('Улучшение качества модели')
for i, v in enumerate(r2_scores):
    ax4.text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.show()

# 8. ФИНАЛЬНЫЕ ВЫВОДЫ
print("\n8. ФИНАЛЬНЫЕ ВЫВОДЫ")
print("=" * 40)

print(f"КАЧЕСТВО МОДЕЛИ: R² = {r2:.4f}")
print(f"УЛУЧШЕНИЕ: {((r2 - 0.2574) / 0.2574 * 100):+.1f}%")

print(f"\nКЛЮЧЕВЫЕ ФАКТОРЫ:")
for i, (_, row) in enumerate(top_5_features.iterrows(), 1):
    corr = X[row['Признак']].corr(y)
    direction = "увеличивает" if corr > 0 else "снижает"
    print(f"{i}. {row['Признак']}")
    print(f"   (важность: {row['Важность']:.3f}, {direction} закредитованность)")

print(f"\nОСНОВНЫЕ ЗАКОНОМЕРНОСТИ:")
print(f"- Региональные особенности: {regional_importance:.1%} влияния")
print(f"- Тип организации: {opf_importance:.1%} влияния") 
print(f"- Финансовые показатели: {financial_importance:.1%} влияния")


print(f"\nСТАТУС: Модель успешно построена и проанализирована")
