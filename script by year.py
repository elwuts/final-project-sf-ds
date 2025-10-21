import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Настройка отображения
plt.style.use('default')
sns.set_palette("husl")

print("═" * 70)
print("АНАЛИЗ: СВЯЗЬ ЧИСЛЕННОСТИ И МАТЕРИАЛЬНОГО БЛАГОСОСТОЯНИЯ")
print("═" * 70)

# Загрузка данных
file_path = r"data\data 1task.xlsx"
df = pd.read_excel(file_path, sheet_name='report3')

print("📊 Загружены данные:")
print(f"   Размер: {df.shape}")
print(f"   Столбцы: {df.columns.tolist()}")

# Преобразование текстовых диапазонов в числовые значения
range_to_midpoint = {
    '0 - 5': 2.5, '6 - 10': 8, '11 - 15': 13,
    '51 - 100': 75.5, '101 - 150': 125.5, '151 - 200': 175.5,
    '201 - 250': 225.5, '251 - 500': 375.5, '501 - 1 000': 750.5
}

df['ссч_число'] = df['ссч'].map(range_to_midpoint)

# Создание групп размера
def create_size_group(ssch_range):
    if ssch_range in ['0 - 5', '6 - 10', '11 - 15']:
        return 'Малые'
    elif ssch_range in ['51 - 100', '101 - 150', '151 - 200', '201 - 250']:
        return 'Средние'
    else:
        return 'Крупные'

df['размер_группа'] = df['ссч'].apply(create_size_group)

# Расчет финансовых показателей
df['финансовый_результат'] = df['доходы'] - df['расходы']
df['доход_на_сотрудника'] = df['доходы'] / df['ссч_число']
df['прибыль_на_сотрудника'] = df['финансовый_результат'] / df['ссч_число']

print(f"\n✅ Данные подготовлены:")
print(f"   Обработано записей: {len(df)}")

# 1. ОСНОВНЫЕ СТАТИСТИКИ ПО ГРУППАМ
print("\n" + "═" * 70)
print("1. ОСНОВНЫЕ СТАТИСТИКИ ПО ГРУППАМ")
print("═" * 70)

stats_by_group = df.groupby('размер_группа').agg({
    'доход_на_сотрудника': ['mean', 'median', 'std', 'count'],
    'прибыль_на_сотрудника': ['mean', 'median', 'std'],
    'финансовый_результат': ['mean', 'median']
}).round(2)

print(stats_by_group)

# 2. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
print("\n" + "═" * 70)
print("2. КОРРЕЛЯЦИЯ МЕЖДУ ЧИСЛЕННОСТЬЮ И ПОКАЗАТЕЛЯМИ")
print("═" * 70)

correlation_income = df['ссч_число'].corr(df['доход_на_сотрудника'])
correlation_profit = df['ссч_число'].corr(df['прибыль_на_сотрудника'])

print(f"📈 Корреляция численность vs доход на сотрудника: {correlation_income:.3f}")
print(f"📈 Корреляция численность vs прибыль на сотрудника: {correlation_profit:.3f}")

# 3. СТАТИСТИЧЕСКИЕ ТЕСТЫ
print("\n" + "═" * 70)
print("3. СТАТИСТИЧЕСКИЕ ТЕСТЫ РАЗЛИЧИЙ МЕЖДУ ГРУППАМИ")
print("═" * 70)

groups = {
    'Малые': df[df['размер_группа'] == 'Малые']['доход_на_сотрудника'],
    'Средние': df[df['размер_группа'] == 'Средние']['доход_на_сотрудника'],
    'Крупные': df[df['размер_группа'] == 'Крупные']['доход_на_сотрудника']
}

# t-тест между малыми и крупными
t_stat, p_value = stats.ttest_ind(groups['Малые'].dropna(), groups['Крупные'].dropna())
print(f"🔬 t-тест Малые vs Крупные (доход на сотрудника):")
print(f"   t-статистика = {t_stat:.3f}, p-value = {p_value:.4f}")

# ANOVA тест между всеми группами
f_stat, p_value_anova = stats.f_oneway(
    groups['Малые'].dropna(), 
    groups['Средние'].dropna(), 
    groups['Крупные'].dropna()
)
print(f"\n🔬 ANOVA тест между всеми группами:")
print(f"   F-статистика = {f_stat:.3f}, p-value = {p_value_anova:.4f}")

# 4. ВИЗУАЛИЗАЦИЯ
print("\n" + "═" * 70)
print("4. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("═" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# График 1: Доход на сотрудника по группам
sns.boxplot(data=df, x='размер_группа', y='доход_на_сотрудника', ax=axes[0,0])
axes[0,0].set_title('Доход на сотрудника по группам организаций')
axes[0,0].set_xlabel('Размер организации')
axes[0,0].set_ylabel('Доход на сотрудника, RUB')

# График 2: Прибыль на сотрудника по группам
sns.boxplot(data=df, x='размер_группа', y='прибыль_на_сотрудника', ax=axes[0,1])
axes[0,1].set_title('Прибыль на сотрудника по группам организаций')
axes[0,1].set_xlabel('Размер организации')
axes[0,1].set_ylabel('Прибыль на сотрудника, RUB')

# График 3: Точечная диаграмма зависимости
sns.scatterplot(data=df, x='ссч_число', y='доход_на_сотрудника', hue='размер_группа', alpha=0.6, ax=axes[1,0])
axes[1,0].set_title('Зависимость дохода на сотрудника от численности')
axes[1,0].set_xlabel('Среднесписочная численность')
axes[1,0].set_ylabel('Доход на сотрудника, RUB')

# График 4: Средние значения по группам
group_means = df.groupby('размер_группа')[['доход_на_сотрудника', 'прибыль_на_сотрудника']].mean()
group_means.plot(kind='bar', ax=axes[1,1])
axes[1,1].set_title('Средние финансовые показатели по группам')
axes[1,1].set_xlabel('Размер организации')
axes[1,1].set_ylabel('Среднее значение, RUB')
axes[1,1].legend(['Доход на сотрудника', 'Прибыль на сотрудника'])

plt.tight_layout()
plt.savefig('анализ_связи_численность_благосостояние.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. ВЫВОДЫ
print("\n" + "═" * 70)
print("5. ВЫВОДЫ ПО ГИПОТЕЗЕ")
print("═" * 70)

print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
print(f"   • Корреляция численность-доход: {correlation_income:.3f}")
print(f"   • Корреляция численность-прибыль: {correlation_profit:.3f}")
print(f"   • p-value различий (Малые vs Крупные): {p_value:.4f}")
print(f"   • p-value ANOVA: {p_value_anova:.4f}")

print("\n🎯 ИНТЕРПРЕТАЦИЯ:")
if p_value < 0.05 and correlation_income > 0:
    print("   ✅ ГИПОТЕЗА ПОДТВЕРЖДАЕТСЯ: Существует связь между численностью и благосостоянием")
    print("   📈 Организации с большей численностью имеют более высокие показатели на сотрудника")
else:
    print("   ❌ ГИПОТЕЗА НЕ ПОДТВЕРЖДАЕТСЯ: Нет статистически значимой связи")
    print("   📊 Численность не является определяющим фактором благосостояния")

print("\n" + "═" * 70)
print("✅ АНАЛИЗ ЗАВЕРШЕН! Результаты сохранены в файл 'анализ_связи_численность_благосостояние.png'")

print("═" * 70)
