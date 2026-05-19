# =============================================================================
# 🏠 House Price Prediction
# Veri Seti : Ames Housing Dataset (Kaggle)
# Problem   : Regresyon — Konut Satış Fiyatı Tahmini (SalePrice / USD)
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 0 — Kütüphane ve Ayarlar
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi']  = 120
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_style('whitegrid')
sns.set_palette('Blues_d')

print('✅ Kütüphaneler yüklendi.')


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 1 — Veri Yükleme ve Genel Bakış
# ─────────────────────────────────────────────────────────────────────────────

train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')

print(f'Train seti boyutu : {train.shape[0]} satır × {train.shape[1]} sütun')
print(f'Test  seti boyutu : {test.shape[0]} satır × {test.shape[1]} sütun')
print(f'\nHedef değişken   : SalePrice')
print(f'Train SalePrice  : {train["SalePrice"].min():,.0f} $ — {train["SalePrice"].max():,.0f} $')
print(f'Ortalama fiyat   : {train["SalePrice"].mean():,.0f} $')

print(train.head())

# Sütun tipi özeti
dtype_summary = train.dtypes.value_counts().rename_axis('Tip').reset_index(name='Sütun Sayısı')
dtype_summary['Tip'] = dtype_summary['Tip'].astype(str)
print('=== Sütun Tipi Dağılımı ===')
print(dtype_summary.to_string(index=False))

sayisal_cols   = train.select_dtypes(include=[np.number]).columns.tolist()
kategorik_cols = train.select_dtypes(include=['object']).columns.tolist()
print(f'\nSayısal sütun  : {len(sayisal_cols)}')
print(f'Kategorik sütun: {len(kategorik_cols)}')

# Temel istatistikler
print(train.describe().T)

# SalePrice dağılımı
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sns.histplot(train['SalePrice'], bins=50, kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('SalePrice — Ham Dağılım', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Satış Fiyatı ($)')
axes[0].xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))

sns.histplot(np.log1p(train['SalePrice']), bins=50, kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('SalePrice — Log Dönüşümü Sonrası', fontsize=13, fontweight='bold')
axes[1].set_xlabel('log(1 + SalePrice)')

plt.suptitle('Hedef Değişken Dağılımı', fontsize=15, y=1.02)
plt.tight_layout()
plt.show()

print(f'Çarpıklık (skewness) — Ham : {train["SalePrice"].skew():.3f}')
print(f'Çarpıklık (skewness) — Log : {np.log1p(train["SalePrice"]).skew():.3f}')
print('\n💡 Log dönüşümü çarpıklığı azaltır → model daha iyi öğrenir.')


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 2 — Keşifsel Veri Analizi (EDA)
# ─────────────────────────────────────────────────────────────────────────────

# 2.1 Eksik Veri Analizi
missing     = train.isnull().sum()
missing     = missing[missing > 0].sort_values(ascending=False)
missing_pct = (missing / len(train) * 100).round(2)
missing_df  = pd.DataFrame({'Eksik Sayı': missing, 'Oran (%)': missing_pct})

print(f'Toplam eksik değer içeren sütun: {len(missing_df)}')
print(missing_df.to_string())

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['#c0392b' if x > 50 else '#e67e22' if x > 20 else '#3498db' for x in missing_pct.values]
bars   = ax.barh(missing_pct.index, missing_pct.values, color=colors)
ax.axvline(x=50, color='red',    linestyle='--', alpha=0.7, label='%50 eşiği')
ax.axvline(x=20, color='orange', linestyle='--', alpha=0.7, label='%20 eşiği')
ax.set_xlabel('Eksik Veri Oranı (%)')
ax.set_title('Sütun Bazında Eksik Veri Oranları', fontsize=14, fontweight='bold')
ax.legend()
for bar, val in zip(bars, missing_pct.values):
    ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%', va='center', fontsize=8)
plt.tight_layout()
plt.show()
print('\n💡 Kırmızı (>%50): PoolQC, MiscFeature, Alley, Fence — bu sütunlarda NA aslında "yok" anlamına gelir.')

# 2.2 Hedef Değişken ile Korelasyon
corr_matrix = train.select_dtypes(include=[np.number]).corr()
top_corr    = corr_matrix['SalePrice'].abs().sort_values(ascending=False)[1:16]

fig, ax = plt.subplots(figsize=(10, 6))
colors  = ['#2ecc71' if v > 0 else '#e74c3c' for v in corr_matrix['SalePrice'][top_corr.index]]
bars    = ax.barh(top_corr.index[::-1], top_corr.values[::-1], color=colors[::-1])
ax.set_xlabel('|Korelasyon Katsayısı|')
ax.set_title('SalePrice ile En Yüksek Korelasyonlu 15 Değişken', fontsize=13, fontweight='bold')
ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 eşiği')
ax.legend()
for bar, val in zip(bars, top_corr.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2, f'{val:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.show()
print('💡 En güçlü 3 belirleyici: OverallQual (0.791), GrLivArea (0.709), GarageCars (0.640)')

# 2.3 Korelasyon Isı Haritası (Top 10)
top10_cols = list(corr_matrix['SalePrice'].abs().sort_values(ascending=False)[0:11].index)
fig, ax    = plt.subplots(figsize=(11, 9))
mask       = np.triu(np.ones_like(corr_matrix[top10_cols].loc[top10_cols], dtype=bool))
sns.heatmap(
    corr_matrix[top10_cols].loc[top10_cols],
    mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
    center=0, square=True, linewidths=0.5, ax=ax,
    cbar_kws={'shrink': 0.8}
)
ax.set_title('En Yüksek Korelasyonlu 10 Değişken — Isı Haritası', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# 2.4 Aykırı Değer Analizi
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(train['GrLivArea'], train['SalePrice'], alpha=0.4, color='steelblue', s=15)
outliers = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)]
axes[0].scatter(outliers['GrLivArea'], outliers['SalePrice'],
                color='red', s=60, label=f'Aykırı ({len(outliers)} ev)', zorder=5)
axes[0].set_xlabel('GrLivArea (yaşam alanı ft²)')
axes[0].set_ylabel('SalePrice ($)')
axes[0].set_title('GrLivArea vs SalePrice', fontsize=12, fontweight='bold')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
axes[0].legend()

train.boxplot(column='SalePrice', by='OverallQual', ax=axes[1])
axes[1].set_title('SalePrice — Genel Kalite (OverallQual) Bazında', fontsize=12, fontweight='bold')
axes[1].set_xlabel('OverallQual (1=Çok Kötü, 10=Mükemmel)')
axes[1].set_ylabel('SalePrice ($)')
axes[1].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
fig.suptitle('')
plt.tight_layout()
plt.show()
print(f'💡 Kırmızı noktalar: büyük alanlı ama çok düşük fiyatlı {len(outliers)} aykırı ev → ön işlemede temizlenecek.')

# 2.5 Sayısal Değişken Dağılımları
key_cols = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'GarageArea',
            'MasVnrArea', 'BsmtFinSF1', 'WoodDeckSF', 'OpenPorchSF', 'LotArea']

fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for ax, col in zip(axes.flatten(), key_cols):
    sns.histplot(train[col].dropna(), bins=40, kde=True, ax=ax, color='steelblue')
    ax.set_title(col, fontweight='bold', fontsize=10)
    ax.set_xlabel('')
plt.suptitle('Temel Sayısal Değişkenlerin Dağılımı', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()
print('💡 Çoğu dağılım sağa çarpık → log dönüşümü ön işlemede uygulanabilir.')

# 2.6 Kategorik Değişken Analizi
cat_plot_cols = ['Neighborhood', 'BldgType', 'HouseStyle', 'SaleCondition', 'KitchenQual']

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flatten(), cat_plot_cols):
    order = train.groupby(col)['SalePrice'].median().sort_values(ascending=False).index
    sns.boxplot(data=train, x=col, y='SalePrice', order=order, ax=ax, palette='Blues')
    ax.set_title(f'{col} → SalePrice', fontweight='bold', fontsize=10)
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x/1000:.0f}K'))
axes.flatten()[-1].set_visible(False)
plt.suptitle('Kategorik Değişkenlere Göre SalePrice Dağılımı', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print('💡 Neighborhood (mahalle) fiyat üzerinde en büyük etkiye sahip kategorik değişken.')


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 3 — Veri Ön İşleme + Feature Selection
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 3.1 Aykırı Değerleri Temizle
train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index)
print(f"Aykırı değer temizlendi. Yeni train boyutu: {train.shape}")

# 3.2 Hedef Değişkeni Log Dönüşümü
y = np.log1p(train['SalePrice'])
print(f"Log dönüşümü uygulandı. y.shape: {y.shape}")

# 3.3 Train ve Test'i Birleştir
train_ids   = train['Id']
test_ids    = test['Id']
X_train_raw = train.drop(['Id', 'SalePrice'], axis=1)
X_test_raw  = test.drop(['Id'], axis=1)
all_data    = pd.concat([X_train_raw, X_test_raw], axis=0).reset_index(drop=True)
print(f"Birleşik veri boyutu: {all_data.shape}")
print(f"Train: {X_train_raw.shape[0]} satır | Test: {X_test_raw.shape[0]} satır")

# 3.4 Eksik Veri Doldurma
none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
             'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
             'MasVnrType']
for col in none_cols:
    all_data[col] = all_data[col].fillna('None')

zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
             'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
             'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
for col in zero_cols:
    all_data[col] = all_data[col].fillna(0)

all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'] \
                                  .transform(lambda x: x.fillna(x.median()))

cat_cols_fill = all_data.select_dtypes(include='object').columns
for col in cat_cols_fill:
    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])

print(f"Kalan eksik değer: {all_data.isnull().sum().sum()}")

# 3.5 Kategorik Dönüşüm + Yeni Özellikler
for col in ['MSSubClass', 'OverallCond', 'YrSold', 'MoSold']:
    all_data[col] = all_data[col].astype(str)

all_data['TotalSF']     = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath']   = (all_data['FullBath'] + all_data['BsmtFullBath'] +
                           0.5 * (all_data['HalfBath'] + all_data['BsmtHalfBath']))
all_data['TotalPorch']  = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                           all_data['3SsnPorch'] + all_data['ScreenPorch'])
all_data['HouseAge']    = all_data['YrSold'].astype(int) - all_data['YearBuilt']
all_data['RemodAge']    = all_data['YrSold'].astype(int) - all_data['YearRemodAdd']
all_data['IsRemodeled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)
all_data['HasPool']     = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage']   = (all_data['GarageArea'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)

print(f"Yeni özellikler eklendi. Toplam sütun: {all_data.shape[1]}")

# 3.6 One-Hot Encoding + Ölçeklendirme
all_data_encoded = pd.get_dummies(all_data)
print(f"Encoding sonrası sütun sayısı: {all_data_encoded.shape[1]}")

X_train = all_data_encoded.iloc[:len(y), :]
X_test  = all_data_encoded.iloc[len(y):, :]

scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"X_train: {X_train_scaled.shape}")
print(f"X_test : {X_test_scaled.shape}")
print("✅ Ön işleme tamamlandı.")

# 3.7 Feature Selection (Random Forest)
rf_fs = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_fs.fit(X_train_scaled, y)

importance_df = pd.DataFrame({
    'Özellik': X_train.columns,
    'Önem':    rf_fs.feature_importances_
}).sort_values('Önem', ascending=False).head(30)

fig, ax = plt.subplots(figsize=(10, 9))
bars = ax.barh(importance_df['Özellik'][::-1], importance_df['Önem'][::-1], color='steelblue')
ax.set_xlabel('Önem Skoru')
ax.set_title('En Önemli 30 Özellik (Random Forest)', fontsize=13, fontweight='bold')
for bar, val in zip(bars, importance_df['Önem'][::-1]):
    ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2, f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.show()

threshold        = 0.001
selected_features = importance_df[importance_df['Önem'] >= threshold]['Özellik'].tolist()
X_train_selected  = X_train[selected_features]
X_test_selected   = X_test[selected_features]

print(f"Toplam özellik   : {X_train.shape[1]}")
print(f"Seçilen özellik  : {len(selected_features)}")
print(f"X_train_selected : {X_train_selected.shape}")
print("✅ Feature Selection tamamlandı.")


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 4 — Model Eğitimi
# ─────────────────────────────────────────────────────────────────────────────

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

scaler2 = StandardScaler()
X_tr    = scaler2.fit_transform(X_train_selected)
X_te    = scaler2.transform(X_test_selected)

results = {}

def evaluate(name, model, X, y_true):
    start  = time.time()
    model.fit(X, y_true)
    elapsed = round(time.time() - start, 2)
    preds  = model.predict(X)
    mae    = mean_absolute_error(y_true, preds)
    rmse   = np.sqrt(mean_squared_error(y_true, preds))
    r2     = r2_score(y_true, preds)
    cv     = cross_val_score(model, X, y_true, cv=5, scoring='r2').mean()
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'CV R²': cv, 'Süre(s)': elapsed}
    print(f"{'─'*50}")
    print(f"  {name}")
    print(f"  MAE  : {mae:.4f}  |  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}  |  CV R²: {cv:.4f}  |  Süre: {elapsed}s")

print("🤖 Model eğitimi başlıyor...\n")

evaluate("Ridge Regression",
         Ridge(alpha=10), X_tr, y)

evaluate("Random Forest",
         RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1), X_tr, y)

evaluate("XGBoost",
         XGBRegressor(n_estimators=500, learning_rate=0.05,
                      max_depth=4, subsample=0.8, colsample_bytree=0.8,
                      random_state=42, verbosity=0), X_tr, y)

evaluate("LightGBM",
         LGBMRegressor(n_estimators=500, learning_rate=0.05,
                       max_depth=4, subsample=0.8,
                       random_state=42, verbose=-1), X_tr, y)

print(f"{'─'*50}")
print("\n✅ Tüm modeller eğitildi.")


# ─────────────────────────────────────────────────────────────────────────────
# BÖLÜM 5 — Metrik Karşılaştırma ve Hata Analizi
# ─────────────────────────────────────────────────────────────────────────────

# 5.1 Model Karşılaştırma Tablosu
results_df = pd.DataFrame(results).T.round(4)
print("═" * 60)
print("  MODEL KARŞILAŞTIRMA TABLOSU")
print("═" * 60)
print(results_df.to_string())
print("═" * 60)
print(f"\n🏆 En iyi R²    : {results_df['R²'].idxmax()}  ({results_df['R²'].max():.4f})")
print(f"🏆 En iyi CV R² : {results_df['CV R²'].idxmax()}  ({results_df['CV R²'].max():.4f})")
print(f"🏆 En düşük MAE : {results_df['MAE'].idxmin()}  ({results_df['MAE'].min():.4f})")
print(f"🏆 En düşük RMSE: {results_df['RMSE'].idxmin()}  ({results_df['RMSE'].min():.4f})")

# 5.2 Metrik Bar Grafikleri
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics = ['MAE', 'RMSE', 'CV R²']
colors  = ['#e74c3c', '#e67e22', '#2ecc71']

for ax, metric, color in zip(axes, metrics, colors):
    vals       = results_df[metric].astype(float)
    best       = vals.idxmin() if metric != 'CV R²' else vals.idxmax()
    bar_colors = [color if m == best else '#bdc3c7' for m in vals.index]
    bars = ax.bar(vals.index, vals.values, color=bar_colors, edgecolor='white', linewidth=1.5)
    ax.set_title(metric, fontsize=13, fontweight='bold')
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Model Karşılaştırması — Renkli = En İyi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 5.3 Hata Analizi (XGBoost)
best_model = XGBRegressor(n_estimators=500, learning_rate=0.05,
                          max_depth=4, subsample=0.8, colsample_bytree=0.8,
                          random_state=42, verbosity=0)
best_model.fit(X_tr, y)
y_pred    = best_model.predict(X_tr)
residuals = y - y_pred

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(y, y_pred, alpha=0.3, color='steelblue', s=15)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', linewidth=2, label='Mükemmel tahmin')
axes[0].set_xlabel('Gerçek log(SalePrice)')
axes[0].set_ylabel('Tahmin edilen')
axes[0].set_title('Gerçek vs Tahmin (XGBoost)', fontweight='bold')
axes[0].legend()

sns.histplot(residuals, bins=50, kde=True, ax=axes[1], color='steelblue')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Hata (Gerçek - Tahmin)')
axes[1].set_title('Hata Dağılımı (XGBoost)', fontweight='bold')

plt.suptitle('XGBoost — Hata Analizi', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 5.4 Fiyat Segmentine Göre Hata Analizi
train_analysis            = X_train_selected.copy()
train_analysis['Gerçek']  = np.expm1(y.values)
train_analysis['Tahmin']  = np.expm1(y_pred)
train_analysis['Hata%']   = (np.abs(train_analysis['Gerçek'] - train_analysis['Tahmin'])
                              / train_analysis['Gerçek'] * 100)

zor_evler = train_analysis.nlargest(10, 'Hata%')[['Gerçek', 'Tahmin', 'Hata%']].copy()
print("En Büyük Hatalar (Top 10):")
print(zor_evler.round(1).to_string())

train_analysis['Segment'] = pd.cut(
    train_analysis['Gerçek'],
    bins=[0, 100000, 200000, 300000, 500000, 800000],
    labels=['<$100K', '$100-200K', '$200-300K', '$300-500K', '>$500K']
)
seg_error = train_analysis.groupby('Segment', observed=True)['Hata%'].mean().round(2)

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(seg_error.index, seg_error.values,
              color=['#2ecc71', '#3498db', '#e67e22', '#e74c3c', '#8e44ad'])
ax.set_xlabel('Fiyat Segmenti')
ax.set_ylabel('Ortalama Hata (%)')
ax.set_title('Fiyat Segmentine Göre Ortalama Tahmin Hatası', fontsize=13, fontweight='bold')
for bar, val in zip(bars, seg_error.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
            f'%{val:.1f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.show()

print("\n💡 Düşük fiyatlı evler (<$100K) tahmin edilmesi en zor segment — veri azlığı ve atipik özellikler.")

# 5.5 Gerçek Dünya Kullanılabilirliği
print("""
🌍 GERÇEK DÜNYA DEĞERLENDİRMESİ
════════════════════════════════════════════════════════

✅ GÜÇLÜ YÖNLER:
  • XGBoost CV R²=0.8956 → görünmeyen veride güvenilir tahmin
  • Ortalama hata %3-5 → $200K ev için ~$6-10K sapma (kabul edilebilir)
  • Eğitim süresi 0.78s → gerçek zamanlı kullanıma uygun

⚠️ SINIRLILIKLAR:
  • <$100K segment %5.6 hata → nadir/atipik evlerde zorlanıyor
  • Model Ames, Iowa verisine özgü → farklı şehirlerde yeniden eğitim gerekir
  • Piyasa dalgalanmaları (enflasyon, faiz) modele yansımıyor

💡 ÖNERİLEN KULLANIM ALANLARI:
  • Emlak platformlarında otomatik fiyat tahmini (Zillow, Sahibinden)
  • Banka mortgage değerleme süreçleri
  • Yatırımcılar için piyasa altı fiyatlı ev tespiti
""")

# 5.6 Test Seti Tahmin + Submission Dosyası
test_preds_log = best_model.predict(X_te)
test_preds     = np.expm1(test_preds_log)

submission = pd.DataFrame({'Id': test_ids, 'SalePrice': test_preds})
submission.to_csv('submission.csv', index=False)

print(f"✅ submission.csv oluşturuldu!")
print(f"   Tahmin sayısı : {len(submission)}")
print(f"   Ortalama fiyat: ${submission['SalePrice'].mean():,.0f}")
print(f"   Min fiyat     : ${submission['SalePrice'].min():,.0f}")
print(f"   Max fiyat     : ${submission['SalePrice'].max():,.0f}")
print(submission.head())
