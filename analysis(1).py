import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# 1) VERİYİ OKUMA
# --------------------------------------------------
print("Veri okunuyor, lütfen bekleyin...\n")

kullanilacak_sutunlar = [
    'EdLevel', 'YearsCodePro', 'AISelect', 'AIAcc',
    'ConvertedCompYearly', 'DevType',
    'LanguageHaveWorkedWith', 'RemoteWork'
]

df = pd.read_csv('survey_results_public.csv', usecols=kullanilacak_sutunlar)

# --------------------------------------------------
# 2) TEMİZLİK VE DÖNÜŞÜMLER
# --------------------------------------------------

# AIAcc hariç gerekli sütunlarda boş olan satırları kaldır
df = df.dropna(subset=[
    'EdLevel', 'YearsCodePro', 'AISelect',
    'ConvertedCompYearly', 'DevType',
    'LanguageHaveWorkedWith', 'RemoteWork'
])

# AI kullanmayanları silmek yerine "No AI" olarak işaretle
df['AIAcc'] = df['AIAcc'].fillna('No AI')


# --- Eğitim seviyesini sayısallaştır ---
def egitim_kodla(deger):
    if any(x in str(deger) for x in ['Bachelor', 'Master', 'Professional']):
        return 1
    return 0

df['EdLevel_Numeric'] = df['EdLevel'].apply(egitim_kodla)


# --- AI kullanımı ---
def ai_kullanim_kodla(deger):
    return 1 if str(deger).startswith('Yes') else 0

df['AISelect_Numeric'] = df['AISelect'].apply(ai_kullanim_kodla)


# --- AI güven skoru ---
ai_guven_haritasi = {
    'Highly distrust': 1,
    'Somewhat distrust': 2,
    'Neither trust nor distrust': 3,
    'Somewhat trust': 4,
    'Highly trust': 5,
    'No AI': 0
}

df['AIAcc_Score'] = df['AIAcc'].map(ai_guven_haritasi)


# --- Deneyim yılı temizleme ---
def deneyim_temizle(deger):
    if deger == 'Less than 1 year':
        return 0.5
    if deger == 'More than 50 years':
        return 50.0
    try:
        return float(deger)
    except:
        return np.nan

df['YearsCodePro'] = df['YearsCodePro'].apply(deneyim_temizle)


# --- Maaş filtresi ---
df = df[
    (df['ConvertedCompYearly'] >= 10000) &
    (df['ConvertedCompYearly'] <= 500000)
]

# Artık kullanmayacağımız sütunları at
df = df.drop(columns=['EdLevel', 'AISelect', 'AIAcc'])

# Son temizlik
df = df.dropna()

print("Temizlik tamamlandı!")
print(f"Kullanılacak veri sayısı: {df.shape[0]}\n")


# --------------------------------------------------
# 3) ÖZET İSTATİSTİK
# --------------------------------------------------
print("--- İSTATİSTİKSEL ÖZET ---")
print(df.describe().round(2))


# --------------------------------------------------
# 4) GRAFİKLER
# --------------------------------------------------
print("\nGrafikler oluşturuluyor...")

sns.set_theme(style="whitegrid")


# 1) Eğitim vs Deneyim
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
print("boxplot.png kaydedildi.")


# 2) AI kullanımı vs Maaş
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
print("barchart.png kaydedildi.")


# 3) Maaş dağılımı
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
print("histogram.png kaydedildi.")


# 4) En yaygın developer rolleri
df['MainBranch'] = df['DevType'].str.split(';').str[0]
en_populer_roller = df['MainBranch'].value_counts().head(5)

plt.figure(figsize=(8, 8))
plt.pie(
    en_populer_roller.values,
    labels=en_populer_roller.index,
    autopct='%1.1f%%',
    startangle=140,
    colors=sns.color_palette('pastel')
)
plt.title('Top 5 Developer Roles')
plt.savefig('piechart.png', dpi=300, bbox_inches='tight')
plt.close()

df = df.drop(columns=['MainBranch'])
print("piechart.png kaydedildi.")


# 5) Dil bazlı AI kullanımı
diller = ['Python', 'Java', 'C++', 'C']
ai_oranlari = []

for dil in diller:
    kullananlar = df[df['LanguageHaveWorkedWith'].str.contains(dil, na=False, regex=False)]
    oran = kullananlar['AISelect_Numeric'].mean() * 100
    ai_oranlari.append(oran)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=ai_oranlari,
    y=diller,
    hue=diller,
    palette='viridis',
    legend=False
)
plt.title('AI Usage Rate by Language (%)')
plt.xlabel('Usage (%)')
plt.ylabel('Language')
plt.savefig('language_barchart.png', dpi=300, bbox_inches='tight')
plt.close()
print("language_barchart.png kaydedildi.")


# 6) Remote vs Maaş
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
print("remote_boxplot.png kaydedildi.")


print("\nTüm işlemler başarıyla tamamlandı!")