import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

columns_to_use = [
    'EdLevel', 'YearsCodePro', 'AISelect', 'AIAcc',
    'ConvertedCompYearly', 'DevType',
    'LanguageHaveWorkedWith', 'RemoteWork'
]

df = pd.read_csv('survey_results_public.csv', usecols=columns_to_use)

df = df.dropna(subset=[
    'EdLevel', 'YearsCodePro', 'AISelect',
    'ConvertedCompYearly', 'DevType',
    'LanguageHaveWorkedWith', 'RemoteWork'
])

df['AIAcc'] = df['AIAcc'].fillna('No AI')

def encode_education(value):
    if any(x in str(value) for x in ['Bachelor', 'Master', 'Professional']):
        return 1
    return 0

df['EdLevel_Numeric'] = df['EdLevel'].apply(encode_education)

def encode_ai_usage(value):
    return 1 if str(value).startswith('Yes') else 0

df['AISelect_Numeric'] = df['AISelect'].apply(encode_ai_usage)

ai_trust_map = {
    'Highly distrust': 1,
    'Somewhat distrust': 2,
    'Neither trust nor distrust': 3,
    'Somewhat trust': 4,
    'Highly trust': 5,
    'No AI': 0
}

df['AIAcc_Score'] = df['AIAcc'].map(ai_trust_map)

def clean_experience(value):
    if value == 'Less than 1 year':
        return 0.5
    if value == 'More than 50 years':
        return 50.0
    try:
        return float(value)
    except ValueError:
        return np.nan

df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

df = df[
    (df['ConvertedCompYearly'] >= 10000) &
    (df['ConvertedCompYearly'] <= 500000)
]

df = df.drop(columns=['EdLevel', 'AISelect', 'AIAcc'])
df = df.dropna()

print(df.describe().round(2))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.boxplot(
    x='EdLevel_Numeric',
    y='YearsCodePro',
    hue='EdLevel_Numeric',
    data=df,
    palette='Set2',
    legend=False
)
plt.title('Professional Experience by Education Level')
plt.xlabel('Education (0: Alternative, 1: University)')
plt.ylabel('Years of Experience')
plt.savefig('boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.barplot(
    x='AISelect_Numeric',
    y='ConvertedCompYearly',
    hue='AISelect_Numeric',
    data=df,
    palette='Set1',
    errorbar=None,
    legend=False
)
plt.title('Average Salary by AI Usage')
plt.xlabel('AI Usage (0: No, 1: Yes)')
plt.ylabel('Annual Salary (USD)')
plt.savefig('barchart.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.histplot(
    df['ConvertedCompYearly'],
    kde=True,
    bins=40,
    color='purple'
)
plt.title('Salary Distribution')
plt.xlabel('Annual Salary (USD)')
plt.ylabel('Frequency')
plt.savefig('histogram.png', dpi=300, bbox_inches='tight')
plt.close()

df['MainBranch'] = df['DevType'].str.split(';').str[0]
top_roles = df['MainBranch'].value_counts().head(5)

plt.figure(figsize=(8, 8))
plt.pie(
    top_roles.values,
    labels=top_roles.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel')
)
plt.title('Top 5 Developer Roles')
plt.savefig('piechart.png', dpi=300, bbox_inches='tight')
plt.close()

df = df.drop(columns=['MainBranch'])

languages = ['Python', 'Java', 'C++', 'C']
ai_rates = []

for lang in languages:
    users = df[df['LanguageHaveWorkedWith'].str.contains(lang, na=False, regex=False)]
    rate = users['AISelect_Numeric'].mean() * 100
    ai_rates.append(rate)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=ai_rates,
    y=languages,
    hue=languages,
    palette='viridis',
    legend=False
)
plt.title('AI Usage Rate by Language (%)')
plt.xlabel('Usage (%)')
plt.ylabel('Language')
plt.savefig('language_barchart.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
sns.boxplot(
    x='RemoteWork',
    y='ConvertedCompYearly',
    hue='RemoteWork',
    data=df,
    palette='Set2',
    legend=False
)
plt.title('Salary by Work Model')
plt.xlabel('Work Type')
plt.ylabel('Annual Salary (USD)')
plt.savefig('remote_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

ai_yes = df[df['AISelect_Numeric'] == 1]['ConvertedCompYearly']
ai_no = df[df['AISelect_Numeric'] == 0]['ConvertedCompYearly']
t_stat, p_val_t = stats.ttest_ind(ai_no, ai_yes, equal_var=False)
print(f"T-Test p-value: {p_val_t:.4e}")

remote = df[df['RemoteWork'] == 'Remote']['ConvertedCompYearly']
hybrid = df[df['RemoteWork'] == 'Hybrid (some remote, some in-person)']['ConvertedCompYearly']
inperson = df[df['RemoteWork'] == 'In-person']['ConvertedCompYearly']
f_stat, p_val_anova = stats.f_oneway(remote, hybrid, inperson)
print(f"ANOVA p-value: {p_val_anova:.4e}")

model = smf.ols('ConvertedCompYearly ~ YearsCodePro + EdLevel_Numeric + AISelect_Numeric + C(RemoteWork)', data=df).fit()
print(model.summary())

conf_int = model.conf_int()
coef_df = pd.DataFrame({
    'coef': model.params,
    'err': model.params - conf_int[0]
}).drop('Intercept')

coef_df.index = ['In-person (vs Hybrid)', 'Remote (vs Hybrid)', 'Years Experience', 'University Degree', 'Uses AI']

plt.figure(figsize=(10, 6))
plt.errorbar(coef_df['coef'], coef_df.index, xerr=coef_df['err'], fmt='o', color='darkblue', 
             ecolor='lightgray', elinewidth=3, capsize=0, markersize=8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.title('Impact of Variables on Annual Salary (OLS Coefficients with 95% CI)')
plt.xlabel('Impact on Salary (USD)')
plt.savefig('coefficient_plot.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(8, 6))
numeric_cols = ['ConvertedCompYearly', 'YearsCodePro', 'EdLevel_Numeric', 'AISelect_Numeric']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numerical Variables')
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 6))
sns.regplot(x='YearsCodePro', y='ConvertedCompYearly', data=df, 
            scatter_kws={'alpha':0.1, 's':15, 'color':'steelblue'}, 
            line_kws={'color':'red', 'linewidth':2})
plt.title('Linear Regression: Experience vs. Salary')
plt.xlabel('Years of Professional Coding')
plt.ylabel('Annual Salary (USD)')
plt.savefig('regression_line.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure(figsize=(9, 6))
sns.scatterplot(x=model.fittedvalues, y=model.resid, alpha=0.15, color='purple', s=15)
plt.axhline(0, color='black', linestyle='--', linewidth=2)
plt.title('Residuals vs. Fitted Values')
plt.xlabel('Predicted Salary (USD)')
plt.ylabel('Residuals (Errors)')
plt.savefig('residual_plot.png', dpi=300, bbox_inches='tight')
plt.close()