import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Загрузка данных
file_path = "\data\data 1task.xlsx"
df = pd.read_excel(file_path, sheet_name='report5')

# Предобработка
df_clean = df.dropna(subset=['ссч примерное', 'доход на сотрудника', 'прибыль на сотрудника'])
df_clean['ссч примерное'] = pd.to_numeric(df_clean['ссч примерное'], errors='coerce')
df_clean = df_clean.dropna(subset=['ссч примерное'])

# Типы организаций для анализа
org_types = ['Благотворительные фонды', 'Общественные фонды', 'Фонды', 
             'Экологические фонды', 'Общественные организации']

print("="*70)
print("АНАЛИЗ ПО ТИПАМ ОРГАНИЗАЦИЙ")
print("="*70)

# Функция для безопасного расчета корреляции
def safe_correlation(x, y):
    """Безопасный расчет корреляции с проверкой на константные значения"""
    # Проверяем, что оба массива не константные и имеют вариацию
    if len(x) < 2 or len(y) < 2:
        return (np.nan, np.nan)
    
    if np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return stats.pearsonr(x, y)
    except:
        return (np.nan, np.nan)

# Создаем фигуру для визуализации
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# Анализ для каждого типа организации
for i, org_type in enumerate(org_types):
    if i >= len(axes) - 1:  # Оставляем последний график для сводки
        break
        
    # Фильтруем данные по типу организации
    org_data = df_clean[df_clean['Организационно-правовая форма'] == org_type]
    
    if len(org_data) > 3:
        # Безопасный расчет корреляций
        corr_income = safe_correlation(org_data['ссч примерное'], org_data['доход на сотрудника'])
        corr_profit = safe_correlation(org_data['ссч примерное'], org_data['прибыль на сотрудника'])
        
        # Визуализация
        income_r = f"r={corr_income[0]:.2f}" if not np.isnan(corr_income[0]) else "r=не опр."
        profit_r = f"r={corr_profit[0]:.2f}" if not np.isnan(corr_profit[0]) else "r=не опр."
        
        axes[i].scatter(org_data['ссч примерное'], org_data['доход на сотрудника'], 
                       alpha=0.7, color='blue', label=f'Доход ({income_r})', s=60)
        axes[i].scatter(org_data['ссч примерное'], org_data['прибыль на сотрудника'], 
                       alpha=0.7, color='red', label=f'Прибыль ({profit_r})', s=60)
        
        axes[i].set_xlabel('Численность работников')
        axes[i].set_ylabel('Финансовые показатели')
        axes[i].set_title(f'{org_type}\n(n={len(org_data)})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Вывод результатов
        print(f"\n{org_type}:")
        if not np.isnan(corr_income[0]):
            print(f"  Доход: r={corr_income[0]:.3f}, p={corr_income[1]:.3f}")
        else:
            print(f"  Доход: невозможно вычислить (нет вариации в данных)")
            
        if not np.isnan(corr_profit[0]):
            print(f"  Прибыль: r={corr_profit[0]:.3f}, p={corr_profit[1]:.3f}")
        else:
            print(f"  Прибыль: невозможно вычислить (нет вариации в данных)")
    else:
        print(f"\n{org_type}: недостаточно данных (n={len(org_data)})")
        axes[i].text(0.5, 0.5, f'Недостаточно данных\n(n={len(org_data)})', 
                    ha='center', va='center', transform=axes[i].transAxes)
        axes[i].set_title(org_type)

# Сводный график - сравнение всех типов
axes[5].axis('off')
summary_text = "СВОДКА ПО ТИПАМ ОРГАНИЗАЦИЙ:\n\n"

for org_type in org_types:
    org_data = df_clean[df_clean['Организационно-правовая форма'] == org_type]
    if len(org_data) > 3:
        corr_income = safe_correlation(org_data['ссч примерное'], org_data['доход на сотрудника'])
        corr_profit = safe_correlation(org_data['ссч примерное'], org_data['прибыль на сотрудника'])
        
        if not np.isnan(corr_income[0]):
            income_sig = "✓" if corr_income[1] < 0.05 else "✗"
            income_str = f"{corr_income[0]:.2f} {income_sig}"
        else:
            income_str = "не опр."
            
        if not np.isnan(corr_profit[0]):
            profit_sig = "✓" if corr_profit[1] < 0.05 else "✗"
            profit_str = f"{corr_profit[0]:.2f} {profit_sig}"
        else:
            profit_str = "не опр."
        
        summary_text += f"{org_type} (n={len(org_data)}):\n"
        summary_text += f"  Доход: {income_str}\n"
        summary_text += f"  Прибыль: {profit_str}\n\n"

axes[5].text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top')
axes[5].set_title('Сводка корреляций\n(✓ - значимо, ✗ - незначимо)')

plt.tight_layout()
plt.show()

# Боксплоты для сравнения распределений
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))

# Подготовка данных для боксплотов
plot_data = []
for org_type in org_types:
    org_subset = df_clean[df_clean['Организационно-правовая форма'] == org_type]
    if len(org_subset) > 0:
        for _, row in org_subset.iterrows():
            plot_data.append({
                'Тип организации': org_type,
                'Численность': row['ссч примерное'],
                'Доход на сотрудника': row['доход на сотрудника'],
                'Прибыль на сотрудника': row['прибыль на сотрудника']
            })

plot_df = pd.DataFrame(plot_data)

if len(plot_df) > 0:
    # Боксплот численности
    sns.boxplot(data=plot_df, x='Тип организации', y='Численность', ax=axes2[0])
    axes2[0].set_title('Численность по типам организаций')
    axes2[0].tick_params(axis='x', rotation=45)
    
    # Боксплот дохода
    sns.boxplot(data=plot_df, x='Тип организации', y='Доход на сотрудника', ax=axes2[1])
    axes2[1].set_title('Доход на сотрудника')
    axes2[1].tick_params(axis='x', rotation=45)
    
    # Боксплот прибыли
    sns.boxplot(data=plot_df, x='Тип организации', y='Прибыль на сотрудника', ax=axes2[2])
    axes2[2].set_title('Прибыль на сотрудника')
    axes2[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# Итоговый вывод
print("\n" + "="*70)
print("ИТОГОВЫЕ ВЫВОДЫ")
print("="*70)

significant_found = False
for org_type in org_types:
    org_data = df_clean[df_clean['Организационно-правовая форма'] == org_type]
    if len(org_data) > 3:
        corr_income = safe_correlation(org_data['ссч примерное'], org_data['доход на сотрудника'])
        corr_profit = safe_correlation(org_data['ссч примерное'], org_data['прибыль на сотрудника'])
        
        if (not np.isnan(corr_income[0]) and corr_income[1] < 0.05) or \
           (not np.isnan(corr_profit[0]) and corr_profit[1] < 0.05):
            significant_found = True
            print(f"\n✓ {org_type}:")
            if not np.isnan(corr_income[0]) and corr_income[1] < 0.05:
                direction = "положительная" if corr_income[0] > 0 else "отрицательная"
                print(f"  Доход: {direction} связь (r={corr_income[0]:.3f})")
            if not np.isnan(corr_profit[0]) and corr_profit[1] < 0.05:
                direction = "положительная" if corr_profit[0] > 0 else "отрицательная"
                print(f"  Прибыль: {direction} связь (r={corr_profit[0]:.3f})")

if not significant_found:
    print("✓ Статистически значимых связей не обнаружено ни в одном типе организаций")

# Общая статистика по типам
print(f"\nОБЩАЯ СТАТИСТИКА:")
for org_type in org_types:
    org_data = df_clean[df_clean['Организационно-правовая форма'] == org_type]
    if len(org_data) > 0:

        print(f"{org_type}: {len(org_data)} организаций")
