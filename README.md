# 🏠 House Price Prediction

**BYM308 Yapay Zekaya Giriş — Grup 40**

Ames Housing Dataset üzerinde makine öğrenimi yöntemleriyle konut satış fiyatı tahmini.


---

## 📋 Proje Özeti

| | |
|---|---|
| **Veri Seti** | [Ames Housing Dataset (Kaggle)](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) |
| **Problem Türü** | Regresyon (Sürekli Değer Tahmini) |
| **Hedef Değişken** | `SalePrice` — Konut Satış Fiyatı (USD) |
| **En İyi Model** | XGBoost — CV R² ≈ 0.8956 |

---

## 📁 Dosya Yapısı

```
├── notebook.py        # Tüm pipeline: EDA → ön işleme → modelleme → analiz
├── train.csv          # Eğitim verisi (Kaggle'dan indirilecek)
├── test.csv           # Test verisi (Kaggle'dan indirilecek)
├── submission.csv     # Model çıktısı — Kaggle'a gönderilecek tahminler
└── README.md
```

> `train.csv` ve `test.csv` dosyalarını [Kaggle yarışma sayfasından](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data) indirip proje klasörüne koyun.

---

## ⚙️ Kurulum

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm
```

Python 3.8+ önerilir.

---

## 🚀 Çalıştırma

```bash
python notebook.py
```

Kod sırasıyla şu adımları gerçekleştirir:

1. Veriyi yükler ve genel istatistikleri gösterir
2. Eksik veri, korelasyon, aykırı değer görselleştirmeleri üretir
3. Eksik değerleri doldurur, yeni özellikler türetir, kodlama ve ölçeklendirme yapar
4. Random Forest ile özellik seçimi yapar
5. Ridge, Random Forest, XGBoost ve LightGBM modellerini eğitir ve karşılaştırır
6. Hata analizi grafikleri üretir
7. `submission.csv` dosyasını oluşturur

---

## 📊 Model Sonuçları

| Model | MAE | RMSE | R² | CV R² |
|---|---|---|---|---|
| Ridge Regression | — | — | — | — |
| Random Forest | — | — | — | — |
| XGBoost | — | — | — | **~0.8956** |
| LightGBM | — | — | — | — |

> Değerler `notebook.py` çalıştırıldığında terminalde görüntülenir.

---

## 🔑 Öne Çıkan Bulgular

- **En güçlü 3 belirleyici:** `OverallQual` (0.791), `GrLivArea` (0.709), `GarageCars` (0.640)
- **Log dönüşümü** hedef değişkenin çarpıklığını ~1.88 → ~0.12'ye indirdi
- **Türetilen özellikler** (`TotalSF`, `TotalBath`, `HouseAge` vb.) model doğruluğunu artırdı
- **En zor segment:** `<$100K` fiyat aralığı — veri azlığı nedeniyle hata oranı daha yüksek

---

## 🌍 Gerçek Dünya Kullanım Alanları

- Emlak platformlarında otomatik fiyat tahmini
- Banka mortgage değerleme süreçleri
- Yatırımcılar için piyasa altı fiyatlı ev tespiti

---

## ⚠️ Sınırlılıklar

- Model Ames, Iowa verisine özgüdür; farklı şehirlerde yeniden eğitim gerekir
- Piyasa dalgalanmaları (enflasyon, faiz oranı) modele yansımamaktadır
