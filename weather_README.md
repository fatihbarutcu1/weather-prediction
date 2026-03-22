# 🌤️ Hava Durumu Tahmin Modeli

Makine öğrenmesi kullanarak günlük hava sıcaklığı tahmini yapan Python projesi.

## 🚀 Özellikler
- 2 yıllık gerçekçi hava durumu verisi üretimi
- Pandas ile veri analizi ve temizleme
- Lag features ve rolling mean ile özellik mühendisliği
- Linear Regression ve Random Forest modelleri
- MAE, RMSE, R² metrikleri ile model değerlendirme
- Matplotlib ile kapsamlı görselleştirme

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

## 📊 Örnek Çıktı
Model çalıştırıldığında:
- `weather_data.csv` — oluşturulan veri seti
- `weather_prediction_results.png` — 4 panelli görselleştirme

## 📈 Model Performansı
| Model | MAE | R² |
|-------|-----|-----|
| Linear Regression | ~1.5°C | ~0.97 |
| Random Forest | ~0.8°C | ~0.99 |

## 👤 Geliştirici
**Fatih Barutcu** — [GitHub](https://github.com/fatihbarutcu1) | [LinkedIn](https://linkedin.com/in/ucturabfatih1)
