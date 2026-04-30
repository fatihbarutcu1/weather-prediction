
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


def load_data(path='weather_data.csv'):
    """
    Open-Meteo'nun CSV formatını okur.
    İlk 3 satır metadata (lat/lon/elevation/timezone), 4. satırdan itibaren asıl veri.
    """
    df = pd.read_csv(path, skiprows=3)

    df.columns = [
        'date', 'temperature', 'temp_max', 'temp_min',
        'humidity', 'wind_speed', 'pressure', 'cloud_cover', 'precipitation'
    ]
    df['date'] = pd.to_datetime(df['date'])
    return df


def add_features(df):
    """Zaman + lag + rolling özellikleri ekler."""
    df = df.copy().sort_values('date').reset_index(drop=True)

    # Zaman özellikleri
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month']       = df['date'].dt.month
    df['year']        = df['date'].dt.year

    # Mevsimsellik için sin/cos
    df['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Geçmiş günlerin sıcaklığı 
    for lag in [1, 2, 3, 7]:
        df[f'temp_lag{lag}'] = df['temperature'].shift(lag)

    
    df['temp_rolling_mean3']  = df['temperature'].shift(1).rolling(3).mean()
    df['temp_rolling_mean7']  = df['temperature'].shift(1).rolling(7).mean()
    df['temp_rolling_mean14'] = df['temperature'].shift(1).rolling(14).mean()


    return df.dropna().reset_index(drop=True)



def train_models(df):
    features = [
        'humidity', 'wind_speed', 'pressure', 'cloud_cover', 'precipitation',
        'sin_doy', 'cos_doy',
        'temp_lag1', 'temp_lag2', 'temp_lag3', 'temp_lag7',
        'temp_rolling_mean3', 'temp_rolling_mean7', 'temp_rolling_mean14',
    ]
    X = df[features]
    y = df['temperature']

    # Time-series için shuffle=False 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)

    # Random Forest 
    rf = RandomForestRegressor(n_estimators=200, max_depth=15,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    return {
        'features': features,
        'X_test': X_test, 'y_test': y_test,
        'lr_pred': lr_pred, 'rf_pred': rf_pred,
        'rf_model': rf,
        'test_dates': df['date'].iloc[-len(y_test):].values,
    }


def evaluate(y_true, y_pred, name):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {name:20s}  MAE={mae:5.2f}°C   RMSE={rmse:5.2f}°C   R²={r2:.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}



def plot_results(df, results, out_path='weather_prediction_results.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Konya Hava Sıcaklığı Tahmin Modeli — 10 Yıllık Gerçek Veri (2015-2024)',
                 fontsize=14, fontweight='bold')

    y_test, rf_pred = results['y_test'], results['rf_pred']
    test_dates = results['test_dates']

    # 1. Test setinde Gerçek vs Tahmin
    ax1 = axes[0, 0]
    ax1.plot(test_dates, y_test.values, label='Gerçek', color='#2563EB', alpha=0.8, linewidth=1.2)
    ax1.plot(test_dates, rf_pred,       label='Tahmin (RF)', color='#F97316', alpha=0.8, linewidth=1.2, linestyle='--')
    ax1.set_title('Test Setinde Gerçek vs Tahmin Sıcaklık')
    ax1.set_ylabel('Sıcaklık (°C)')
    ax1.legend(loc='upper right')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax1.tick_params(axis='x', rotation=30)
    ax1.grid(alpha=0.3)

    # 2. 10 yıllık trend
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['temperature'], color='#2563EB', alpha=0.4, linewidth=0.5, label='Günlük')
    ax2.plot(df['date'], df['temp_rolling_mean14'], color='#EF4444', linewidth=1.5, label='14 Günlük Ort.')
    ax2.set_title('10 Yıllık Sıcaklık Trendi (Konya)')
    ax2.set_ylabel('Sıcaklık (°C)')
    ax2.legend(loc='upper right')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(alpha=0.3)

    # 3. Özellik
    ax3 = axes[1, 0]
    importances = pd.Series(results['rf_model'].feature_importances_,
                            index=results['features']).sort_values()
    colors = ['#94A3B8'] * (len(importances) - 5) + ['#2563EB'] * 5  # En önemli 5 mavi
    importances.plot(kind='barh', ax=ax3, color=colors)
    ax3.set_title('Özellik Önemi (Random Forest)')
    ax3.set_xlabel('Önem Skoru')
    ax3.grid(alpha=0.3, axis='x')

    # 4. Scatter Gerçek vs Tahmin
    ax4 = axes[1, 1]
    ax4.scatter(y_test, rf_pred, alpha=0.4, color='#2563EB', s=10)
    lims = [min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())]
    ax4.plot(lims, lims, 'r--', linewidth=1.5, label='Mükemmel Tahmin')
    r2 = r2_score(y_test, rf_pred)
    mae = mean_absolute_error(y_test, rf_pred)
    ax4.set_xlabel('Gerçek Sıcaklık (°C)')
    ax4.set_ylabel('Tahmin Edilen Sıcaklık (°C)')
    ax4.set_title(f'Tahmin Doğruluğu  (R²={r2:.3f},  MAE={mae:.2f}°C)')
    ax4.legend(loc='upper left')
    ax4.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n Grafik kaydedildi: {out_path}")



def main():
    print("Veri yükleniyor...")
    df = load_data('weather_data.csv')
    print(f"   {len(df):,} günlük veri yüklendi ({df['date'].min().date()} → {df['date'].max().date()})\n")

    print("Özellik mühendisliği...")
    df = add_features(df)
    print(f"   {len(df):,} satır kullanılabilir (lag'ler için ilk birkaç satır düştü)\n")

    print("Modeller eğitiliyor...")
    results = train_models(df)
    print()

    print("MODEL SONUÇLARI")
    print("─" * 60)
    evaluate(results['y_test'], results['lr_pred'], 'Linear Regression')
    evaluate(results['y_test'], results['rf_pred'], 'Random Forest')
    print()

    print("Görselleştirme oluşturuluyor...")
    plot_results(df, results)
    print("\n Tamamlandı!")


if __name__ == '__main__':
    main()