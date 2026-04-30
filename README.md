# 🌤️ Konya Hava Sıcaklığı Tahmin Modeli

Open-Meteo Archive API'den çekilmiş **10 yıllık (2015-2024) gerçek günlük hava verisi** kullanarak Konya için yarının sıcaklığını tahmin eden makine öğrenmesi projesi.

## 📊 Sonuçlar

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Linear Regression | **1.15°C** | 1.57°C | 0.969 |
| Random Forest | 1.18°C | 1.59°C | 0.968 |

> Test seti: son 2 yıl (2023-2024). Eğitim seti: ilk 8 yıl (2015-2022).
> Ortalama hata 1.15°C — model "yarın 18°C olacak" derse, gerçek değer tipik olarak 17-19°C arasında çıkar.

## 🎯 Veri Kaynağı

- **API:** [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api) (ücretsiz, key gerektirmez)
- **Konum:** Konya, Türkiye (37.85°N, 32.45°E, 1020m rakım)
- **Aralık:** 1 Ocak 2015 — 31 Aralık 2024
- **Toplam kayıt:** 3.653 gün (eksik veri: 0)
- **Değişkenler:** Sıcaklık (ort/min/max), nem, rüzgar hızı, basınç, bulut örtüsü, yağış

## 🔧 Yöntem

### Özellik Mühendisliği
- **Lag features:** Geçmiş 1, 2, 3 ve 7 günlük sıcaklıklar
- **Rolling means:** 3, 7 ve 14 günlük hareketli ortalamalar (data leakage önleme için `shift(1)` ile)
- **Mevsimsellik:** `sin(2π·gün/365)` ve `cos(2π·gün/365)` ile çevrimsel kodlama
- **Diğer:** Nem, rüzgar, basınç, bulut örtüsü, yağış

### Model Pipeline
1. CSV okuma → kolon isimlerini sadeleştirme
2. Özellik mühendisliği (lag + rolling + sin/cos)
3. Train/test split: time-series için `shuffle=False` (geçmiş→gelecek sırası korunur)
4. Linear Regression (StandardScaler ile) ve Random Forest (200 ağaç, max_depth=15)
5. MAE, RMSE, R² ile değerlendirme

## 🔍 Bulgular

**`temp_lag1` (dünkü sıcaklık) tek başına özellik öneminin ~%97'sini oluşturuyor.** Bu, meteorolojide **persistence forecast** olarak bilinen "yarın bugüne benzer olacak" varsayımının ne kadar güçlü olduğunu gösteriyor — meteorologların yeni modelleri test ederken kullandığı baseline budur.

Linear Regression ve Random Forest'ın neredeyse aynı sonucu vermesi, problemin lineer doğasını gösteriyor: yarının sıcaklığı, son birkaç günün ortalamasının lineer bir kombinasyonu olarak yaklaşık olarak ifade edilebiliyor.

## 🛠️ Kullanılan Teknolojiler

- **Python 3.x**
- **Pandas** — veri işleme
- **NumPy** — sayısal hesaplama
- **Scikit-learn** — makine öğrenmesi
- **Matplotlib** — görselleştirme

## ⚙️ Kurulum ve Çalıştırma

```bash
# Repoyu klonla
git clone https://github.com/fatihbarutcu1/weather-prediction.git
cd weather-prediction

# Gereksinimleri yükle
pip install -r requirements.txt

# Modeli çalıştır
python weather_prediction.py
```

Veri seti `weather_data.csv` zaten repoda mevcut. Yeniden çekmek istersen:
```
https://archive-api.open-meteo.com/v1/archive?latitude=37.8746&longitude=32.4932&start_date=2015-01-01&end_date=2024-12-31&daily=temperature_2m_mean,temperature_2m_max,temperature_2m_min,relative_humidity_2m_mean,wind_speed_10m_max,surface_pressure_mean,cloud_cover_mean,precipitation_sum&timezone=Europe%2FIstanbul&format=csv
```

## 📁 Proje Yapısı

```
weather-prediction/
├── weather_prediction.py          # Ana model kodu (modüler fonksiyonlar)
├── weather_data.csv               # Open-Meteo'dan çekilmiş 10 yıllık veri
├── weather_prediction_results.png # 4 panelli görselleştirme çıktısı
├── requirements.txt
└── README.md
```

## 📈 Çıktı Görseli

`weather_prediction_results.png` 4 panelden oluşur:
1. **Test setinde gerçek vs tahmin** — modelin son 2 yıldaki performansı
2. **10 yıllık sıcaklık trendi** — ham veri + 14 günlük hareketli ortalama
3. **Özellik önemi** — Random Forest hangi değişkenlere ne kadar ağırlık veriyor
4. **Tahmin doğruluğu scatter** — gerçek vs tahmin saçılım grafiği (45° ideal çizgisi ile)

## 🚀 Geliştirilebilecek Noktalar

- Daha uzun vadeli tahmin (`t+3`, `t+7`) — `temp_lag1`'in dominantlığı bu durumda azalır, diğer özellikler ön plana çıkar
- XGBoost veya LightGBM ile model çeşitlendirme
- LSTM ile sıralı (sequence) yaklaşım
- Streamlit ile interaktif web arayüzü
- Çoklu şehir karşılaştırması

## 👤 Geliştirici

**Fatih Barutcu** — [GitHub](https://github.com/fatihbarutcu1) | [LinkedIn](https://www.linkedin.com/in/fatihbarutcu1)
