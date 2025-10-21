import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройки отображения
plt.style.use('default')
sns.set_palette("husl")

print("═" * 70)
print("АНАЛИЗ: РЕГИОНАЛЬНЫЕ РАЗЛИЧИЯ СВЯЗИ ЧИСЛЕННОСТИ И БЛАГОСОСТОЯНИЯ")
print("═" * 70)

# Загрузка данных
file_path = r"E:\SKILLFACTORY\SF\dataSF\finalproject\data\data 1task.xlsx"
df = pd.read_excel(file_path, sheet_name='report4')

print("📊 Загружены данные:")
print(f"   Размер: {df.shape}")
print(f"   Столбцы: {df.columns.tolist()}")

# Проверяем наличие регионов
print(f"\n🌍 Регионы в данных:")
print(f"   Уникальных регионов: {df['регион'].nunique()}")
print(f"   Топ-10 регионов по количеству организаций:")
print(df['регион'].value_counts().head(10))

# 1. ОБЩАЯ СТАТИСТИКА ПО РЕГИОНАМ
print("\n" + "═" * 70)
print("1. ОБЩАЯ СТАТИСТИКА ПО РЕГИОНАМ")
print("═" * 70)

regional_stats = df.groupby('регион').agg({
    'доход на сотрудника': ['mean', 'median', 'std', 'count'],
    'прибыль на сотрудника': ['mean', 'median'],
    'ссч(примерное)': 'mean',
    'финансовый результат': 'mean'
}).round(2)

# Регионы с наибольшим доходом на сотрудника
top_regions_income = regional_stats[('доход на сотрудника', 'mean')].sort_values(ascending=False).head(10)
print("📈 Топ-10 регионов по доходу на сотрудника:")
for region, income in top_regions_income.items():
    print(f"   {region}: {income:,.0f} RUB")

# 2. КОРРЕЛЯЦИЯ ПО РЕГИОНАМ
print("\n" + "═" * 70)
print("2. КОРРЕЛЯЦИЯ ЧИСЛЕННОСТЬ-БЛАГОСОСТОЯНИЕ ПО РЕГИОНАМ")
print("═" * 70)

# Считаем корреляцию для каждого региона
correlation_by_region = {}

for region in df['регион'].unique():
    region_data = df[df['регион'] == region]
    if len(region_data) > 10:  # Только регионы с достаточным количеством данных
        # Убираем пропуски и бесконечные значения
        temp_data = region_data[['ссч(примерное)', 'доход на сотрудника', 'прибыль на сотрудника']].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(temp_data) > 5:
            corr_income = temp_data['ссч(примерное)'].corr(temp_data['доход на сотрудника'])
            corr_profit = temp_data['ссч(примерное)'].corr(temp_data['прибыль на сотрудника'])
            correlation_by_region[region] = {
                'корреляция_доход': corr_income,
                'корреляция_прибыль': corr_profit,
                'количество_организаций': len(temp_data)
            }

# Создаем DataFrame с результатами корреляций
if correlation_by_region:
    corr_df = pd.DataFrame(correlation_by_region).T
    corr_df = corr_df.sort_values('корреляция_доход', ascending=False)

    print("🔍 Регионы с самой сильной положительной корреляцией:")
    strong_positive = corr_df[corr_df['корреляция_доход'] > 0.3].head(10)
    if not strong_positive.empty:
        for region, data in strong_positive.iterrows():
            print(f"   {region}: {data['корреляция_доход']:.3f} (орг: {data['количество_организаций']})")
    else:
        print("   Нет регионов с сильной положительной корреляцией")

    print("\n🔍 Регионы с самой сильной отрицательной корреляцией:")
    strong_negative = corr_df[corr_df['корреляция_доход'] < -0.3].head(10)
    if not strong_negative.empty:
        for region, data in strong_negative.iterrows():
            print(f"   {region}: {data['корреляция_доход']:.3f} (орг: {data['количество_организаций']})")
    else:
        print("   Нет регионов с сильной отрицательной корреляцией")
else:
    print("   Не удалось рассчитать корреляции по регионам")

# 3. СТАТИСТИЧЕСКИЕ ТЕСТЫ ПО РЕГИОНАМ
print("\n" + "═" * 70)
print("3. СТАТИСТИЧЕСКАЯ ЗНАЧИМОСТЬ КОРРЕЛЯЦИЙ ПО РЕГИОНАМ")
print("═" * 70)

print("📊 Регионы со статистически значимой корреляцией (p-value < 0.05):")

significant_regions = []
for region in df['регион'].unique():
    region_data = df[df['регион'] == region]
    if len(region_data) > 20:  # Только для регионов с достаточными данными
        # Очищаем данные
        clean_data = region_data[['ссч(примерное)', 'доход на сотрудника']].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(clean_data) > 10:
            try:
                corr, p_value = stats.pearsonr(
                    clean_data['ссч(примерное)'],
                    clean_data['доход на сотрудника']
                )
                if p_value < 0.05 and abs(corr) > 0.2:
                    significant_regions.append((region, corr, p_value, len(clean_data)))
            except:
                pass

# Сортируем по силе корреляции
significant_regions.sort(key=lambda x: abs(x[1]), reverse=True)

if significant_regions:
    for region, corr, p_val, count in significant_regions[:10]:
        significance = "✅ ПОЛОЖИТЕЛЬНАЯ" if corr > 0 else "❌ ОТРИЦАТЕЛЬНАЯ"
        print(f"   {region}: {significance} r={corr:.3f}, p={p_val:.4f} (n={count})")
else:
    print("   ❌ Нет регионов со статистически значимой корреляцией")

# 4. ВИЗУАЛИЗАЦИЯ
print("\n" + "═" * 70)
print("4. ВИЗУАЛИЗАЦИЯ РЕГИОНАЛЬНЫХ РАЗЛИЧИЙ")
print("═" * 70)

# Выбираем топ-15 регионов по количеству организаций для визуализации
top_regions = df['регион'].value_counts().head(15).index

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# График 1: Доход на сотрудника по регионам
region_income = df[df['регион'].isin(top_regions)].groupby('регион')['доход на сотрудника'].mean().sort_values(ascending=False)
region_income.plot(kind='bar', ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Средний доход на сотрудника по регионам (Топ-15)')
axes[0,0].set_ylabel('Доход на сотрудника, RUB')
axes[0,0].tick_params(axis='x', rotation=45)

# График 2: Корреляции по регионам
if 'corr_df' in locals() and not corr_df.empty:
    top_corr_regions = corr_df.head(10)
    top_corr_regions['корреляция_доход'].plot(kind='bar', ax=axes[0,1], color='lightcoral')
    axes[0,1].set_title('Сила корреляции численность-доход по регионам (Топ-10)')
    axes[0,1].set_ylabel('Коэффициент корреляции')
    axes[0,1].tick_params(axis='x', rotation=45)

# График 3: Точечные диаграммы для регионов с самой сильной корреляцией
if significant_regions:
    strongest_region = significant_regions[0][0]
    region_data = df[df['регион'] == strongest_region]
    
    axes[1,0].scatter(region_data['ссч(примерное)'], region_data['доход на сотрудника'], alpha=0.6)
    axes[1,0].set_title(f'Зависимость в регионе: {strongest_region}\n(корреляция: {significant_regions[0][1]:.3f})')
    axes[1,0].set_xlabel('Среднесписочная численность')
    axes[1,0].set_ylabel('Доход на сотрудника, RUB')

# График 4: Распределение доходов по группам размеров в регионах
if len(top_regions) >= 3:
    sample_regions = list(top_regions[:3])
    sample_data = df[df['регион'].isin(sample_regions)]
    
    sns.boxplot(data=sample_data, x='регион', y='доход на сотрудника', ax=axes[1,1])
    axes[1,1].set_title('Распределение дохода на сотрудника по регионам')
    axes[1,1].set_ylabel('Доход на сотрудника, RUB')
    axes[1,1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('региональный_анализ_связи.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. ВЫВОДЫ
print("\n" + "═" * 70)
print("5. ОСНОВНЫЕ ВЫВОДЫ ПО РЕГИОНАЛЬНОМУ АНАЛИЗУ")
print("═" * 70)

if significant_regions:
    strongest_positive = max(significant_regions, key=lambda x: x[1])
    strongest_negative = min(significant_regions, key=lambda x: x[1])
    
    print(f"🎯 САМАЯ СИЛЬНАЯ ПОЛОЖИТЕЛЬНАЯ СВЯЗЬ:")
    print(f"   📍 {strongest_positive[0]}: r={strongest_positive[1]:.3f}")
    print(f"   💡 В этом регионе большие организации действительно богаче")
    
    print(f"\n🎯 САМАЯ СИЛЬНАЯ ОТРИЦАТЕЛЬНАЯ СВЯЗЬ:")
    print(f"   📍 {strongest_negative[0]}: r={strongest_negative[1]:.3f}")
    print(f"   💡 В этом регионе малые организации эффективнее крупных")
    
    print(f"\n📊 Всего регионов со значимой связью: {len(significant_regions)}")
else:
    print("❌ Нет статистически значимых региональных различий в связи")
    print("💡 Связь численность-благосостояние одинакова во всех регионах")

print("\n" + "═" * 70)
print("✅ РЕГИОНАЛЬНЫЙ АНАЛИЗ ЗАВЕРШЕН!")
print("═" * 70)