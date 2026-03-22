import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. VERİ OLUŞTURMA (Gerçekçi hava durumu simülasyonu)
# ============================================================
np.random.seed(42)
n = 730  # 2 yıllık günlük veri

dates = pd.date_range(start='2022-01-01', periods=n, freq='D')
day_of_year = np.arange(n) % 365

# Mevsimsel sıcaklık eğrisi (Türkiye iklimine uygun)
base_temp = 13 + 18 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
noise = np.random.normal(0, 3, n)
temperature = base_temp + noise

# İlişkili özellikler
humidity    = 70 - 0.4 * temperature + np.random.normal(0, 8, n)
humidity    = np.clip(humidity, 20, 100)
wind_speed  = np.abs(np.random.normal(15, 6, n))
pressure    = 1013 - 0.3 * temperature + np.random.normal(0, 5, n)
cloud_cover = np.clip(40 + 0.3 * humidity + np.random.normal(0, 15, n), 0, 100)

df = pd.DataFrame({
    'date':        dates,
    'temperature': temperature.round(1),
    'humidity':    humidity.round(1),
    'wind_speed':  wind_speed.round(1),
    'pressure':    pressure.round(1),
    'cloud_cover': cloud_cover.round(1),
    'day_of_year': day_of_year,
    'month':       dates.month,
})

df.to_csv('weather_data.csv', index=False)
print("✅ Veri seti oluşturuldu: weather_data.csv")
print(f"   Toplam kayıt: {len(df)}")
print(f"   Sütunlar: {list(df.columns)}\n")

# ============================================================
# 2. VERİ ANALİZİ
# ============================================================
print("📊 VERİ ANALİZİ")
print("-" * 40)
print(df.describe().round(2))
print(f"\nEksik veri: {df.isnull().sum().sum()}")

# ============================================================
# 3. ÖZELLİK MÜHENDİSLİĞİ
# ============================================================
# Geçmiş günlerin sıcaklıklarını özellik olarak ekle (lag features)
df['temp_lag1'] = df['temperature'].shift(1)
df['temp_lag2'] = df['temperature'].shift(2)
df['temp_lag3'] = df['temperature'].shift(3)
df['temp_rolling_mean7'] = df['temperature'].rolling(7).mean()
df = df.dropna()

features = ['humidity', 'wind_speed', 'pressure', 'cloud_cover',
            'day_of_year', 'month', 'temp_lag1', 'temp_lag2',
            'temp_lag3', 'temp_rolling_mean7']

X = df[features]
y = df['temperature']

# ============================================================
# 4. MODEL EĞİTİMİ
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
lr_pred = lr.predict(X_test_sc)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ============================================================
# 5. MODEL DEĞERLENDİRME
# ============================================================
print("\n📈 MODEL SONUÇLARI")
print("-" * 40)
for name, pred in [("Linear Regression", lr_pred), ("Random Forest", rf_pred)]:
    mae  = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2   = r2_score(y_test, pred)
    print(f"{name}:")
    print(f"  MAE  = {mae:.2f}°C   RMSE = {rmse:.2f}°C   R² = {r2:.4f}")

# ============================================================
# 6. GÖRSELLEŞTİRME
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hava Durumu Tahmin Modeli — Fatih Barutcu', fontsize=14, fontweight='bold')

# 1. Gerçek vs Tahmin (RF)
ax1 = axes[0, 0]
test_dates = df['date'].iloc[-len(y_test):]
ax1.plot(test_dates, y_test.values, label='Gerçek', color='#2563EB', alpha=0.8, linewidth=1.5)
ax1.plot(test_dates, rf_pred,       label='Tahmin (RF)', color='#F97316', alpha=0.8, linewidth=1.5, linestyle='--')
ax1.set_title('Gerçek vs Tahmin Sıcaklık')
ax1.set_ylabel('Sıcaklık (°C)')
ax1.legend()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax1.tick_params(axis='x', rotation=30)

# 2. Tüm dönem sıcaklık trendi
ax2 = axes[0, 1]
ax2.plot(df['date'], df['temperature'], color='#2563EB', alpha=0.6, linewidth=0.8)
ax2.plot(df['date'], df['temp_rolling_mean7'], color='#EF4444', linewidth=2, label='7 Günlük Ort.')
ax2.set_title('2 Yıllık Sıcaklık Trendi')
ax2.set_ylabel('Sıcaklık (°C)')
ax2.legend()
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.tick_params(axis='x', rotation=30)

# 3. Özellik Önemi (RF)
ax3 = axes[1, 0]
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
importances.plot(kind='barh', ax=ax3, color='#2563EB')
ax3.set_title('Özellik Önemi (Random Forest)')
ax3.set_xlabel('Önem Skoru')

# 4. Scatter: Gerçek vs Tahmin
ax4 = axes[1, 1]
ax4.scatter(y_test, rf_pred, alpha=0.5, color='#2563EB', s=15)
lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Mükemmel Tahmin')
ax4.set_xlabel('Gerçek Sıcaklık (°C)')
ax4.set_ylabel('Tahmin Edilen (°C)')
ax4.set_title(f'Tahmin Doğruluğu (R²={r2_score(y_test, rf_pred):.3f})')
ax4.legend()

plt.tight_layout()
plt.savefig('weather_prediction_results.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Grafik kaydedildi: weather_prediction_results.png")
