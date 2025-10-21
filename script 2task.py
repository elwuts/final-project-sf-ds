import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import stats

file_path = r'E:\SKILLFACTORY\SF\dataSF\finalproject\data\data_cleaned_panel.xlsx'
df = pd.read_excel(file_path, sheet_name='Sheet1')

print("=== АНАЛИЗ ФАКТОРОВ ВЛИЯНИЯ НА ЗАКРЕДИТОВАННОСТЬ ===")

# 1. ПОДГОТОВКА ДАННЫХ
print("1. ПОДГОТОВКА ДАННЫХ")

# Создаем финансовые показатели
financial_ratios = {
    'рентабельность': lambda x: x['Чистая прибыль (убыток), RUB'] / (x['Выручка, RUB'] + 1),
    'выручка_на_сотрудника': lambda x: x['Выручка, RUB'] / (x['ссч_числовая'] + 1),
    'капитал_к_выручке': lambda x: x['Капитал и резервы, RUB'] / (x['Выручка, RUB'] + 1),
    'доля_кредиторки': lambda x: x['Кредиторская задолженность, RUB'] / (x['Выручка, RUB'] + 1),
    'прибыль_на_сотрудника': lambda x: x['Чистая прибыль (убыток), RUB'] / (x['ссч_числовая'] + 1)
}

for name, func in financial_ratios.items():
    df[name] = func(df)
    df[name] = df[name].replace([np.inf, -np.inf], np.nan)
    df[name] = df[name].fillna(df[name].median())

# Формируем признаки
base_features = [
    'log_выручка', 'log_капитал', 'Чистая прибыль (убыток), RUB',
    'Денежные средства и денежные эквиваленты, RUB', 'Кредиторская задолженность, RUB',
    'ссч_числовая', 'год'
] + list(financial_ratios.keys())

region_cols = [col for col in df.columns if col.startswith('регион_')][:10]
base_features.extend(region_cols)

opf_cols = [col for col in df.columns if col.startswith('опф_')]
base_features.extend(opf_cols)

X = df[base_features].copy()
y = df['log_заемные'].copy()
X = X.fillna(X.median())

print(f"Данные: {X.shape[0]} наблюдений, {X.shape[1]} признаков")

# 2. ПОСТРОЕНИЕ МОДЕЛИ
print("\n2. ПОСТРОЕНИЕ МОДЕЛИ")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"R² модели: {r2:.4f}")

# 3. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ
print("\n3. ВАЖНОСТЬ ПРИЗНАКОВ")

importance_df = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)

print("Топ-10 самых важных признаков:")
for i, row in importance_df.head(10).iterrows():
    correlation = X[row['Признак']].corr(y)
    direction = "↑" if correlation > 0 else "↓"
    print(f"  {row['Признак']:35} : {row['Важность']:.4f} {direction}")

# 4. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ
print("\n4. ПРОВЕРКА СТАТИСТИЧЕСКОЙ ЗНАЧИМОСТИ")

significant_features = []
alpha = 0.05  # уровень значимости

print("Статистически значимые факторы (p-value < 0.05):")
for feature in importance_df.head(10)['Признак']:
    correlation, p_value = stats.pearsonr(X[feature], y)
    if p_value < alpha:
        significant_features.append(feature)
        print(f"  {feature:35} : p-value = {p_value:.6f} (значим)")
    else:
        print(f"  {feature:35} : p-value = {p_value:.6f}")

# 5. ВИЗУАЛИЗАЦИЯ
plt.figure(figsize=(12, 6))

# График важности признаков
plt.subplot(1, 2, 1)
top_10 = importance_df.head(10)
sns.barplot(data=top_10, x='Важность', y='Признак')
plt.title('Топ-10 факторов влияния')

# График качества модели
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Реальные значения')
plt.ylabel('Предсказания')
plt.title(f'Качество модели (R² = {r2:.3f})')

plt.tight_layout()
plt.show()

# 6. ОТВЕТ НА ИССЛЕДОВАТЕЛЬСКИЙ ВОПРОС
print("\n5. ОТВЕТ НА ИССЛЕДОВАТЕЛЬСКИЙ ВОПРОС")
print("=" * 50)

print(f"ГИПОТЕЗА H1: Существуют факторы, имеющие статистически значимую")
print(f"взаимосвязь с уровнем закредитованности компаний")
print()

if significant_features:
    print(f"✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА")
    print(f"Найдено {len(significant_features)} статистически значимых факторов:")
    for feature in significant_features[:5]:  # показываем топ-5
        importance = importance_df[importance_df['Признак'] == feature]['Важность'].values[0]
        correlation = X[feature].corr(y)
        print(f"  - {feature} (важность: {importance:.3f}, корреляция: {correlation:.3f})")
else:
    print(f"❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДЕНА")
    print("Статистически значимых факторов не обнаружено")

print(f"\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
print(f"Качество модели (R²): {r2:.4f}")
print(f"Всего проанализировано признаков: {len(importance_df)}")
print(f"Признаков с важностью > 0.01: {len(importance_df[importance_df['Важность'] > 0.01])}")
print(f"Признаков с важностью > 0.05: {len(importance_df[importance_df['Важность'] > 0.05])}")

print(f"\nВЫВОД: Модель показывает наличие значимых факторов влияния")
print(f"на закредитованность компаний.")