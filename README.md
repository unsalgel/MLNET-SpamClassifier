# ML.NET Spam Mail Sınıflandırıcı

Bu proje, ML.NET kullanarak spam e-posta sınıflandırması yapan bir makine öğrenmesi uygulamasıdır.

## 🎯 Proje Amacı

Bu projenin temel amacı **ML.NET öğrenmek** ve makine öğrenmesi temellerini uygulamalı olarak anlamaktır. Spam mail sınıflandırması, makine öğrenmesine başlamak için ideal bir problemdir çünkü:

- Anlaşılması kolay bir problem
- İyi sonuçlar veren bir problem
- ML.NET'in temel özelliklerini öğrenmek için uygun

## 📊 Veri Seti

- **Kaynak**: [Kaggle](https://www.kaggle.com/)
- **Dosya**: `spam.csv`
- **Toplam Veri**: 5,574 satır
- **Sütunlar**:
  - `v1`: Etiket (ham/spam)
  - `v2`: E-posta mesajı içeriği

Veri seti Kaggle'dan indirilmiştir ve proje içinde `Data/spam.csv` konumunda bulunmaktadır.

## 🚀 Proje Yapısı

```
ML.NET-SpamMailTespit/
├── SpamClassifier/
│   ├── Data/
│   │   └── spam.csv          # Veri seti
│   ├── Program.cs             # Ana program
│   ├── SpamData.cs            # Veri modeli
│   ├── SpamPrediction.cs      # Tahmin modeli
│   └── SpamClassifier.csproj  # Proje dosyası
└── README.md
```

## 🛠️ Teknolojiler

- **.NET 8.0**
- **ML.NET 5.0**
- **C#**

## 📦 Kurulum

1. Projeyi klonlayın veya indirin
2. .NET 8.0 SDK'nın yüklü olduğundan emin olun
3. Proje dizinine gidin:
   ```bash
   cd SpamClassifier
   ```
4. Bağımlılıkları yükleyin:
   ```bash
   dotnet restore
   ```
5. Projeyi çalıştırın:
   ```bash
   dotnet run
   ```

## 🔄 Nasıl Çalışır?

### 1. Veri Yükleme

CSV dosyasından veri yüklenir ve ML.NET'in `IDataView` formatına dönüştürülür.

### 2. Veri Ayırma

Veri seti %80 eğitim ve %20 test olarak ayrılır:

- **Eğitim Seti**: Modeli eğitmek için kullanılır
- **Test Seti**: Modelin performansını ölçmek için kullanılır

### 3. Pipeline Oluşturma

ML.NET pipeline'ı şu adımlardan oluşur:

1. **MapValue**: String label'ları ("ham"/"spam") Boolean değerlere dönüştürür
2. **FeaturizeText**: Mesaj metnini sayısal özelliklere dönüştürür
3. **SdcaLogisticRegression**: Binary sınıflandırma algoritması ile model eğitilir

### 4. Model Eğitimi

Eğitim verisi kullanılarak model eğitilir. Model, spam ve ham mesajları ayırt etmeyi öğrenir.

### 5. Model Değerlendirme

Test verisi kullanılarak modelin performansı ölçülür. Aşağıdaki metrikler hesaplanır:

- **Doğruluk (Accuracy)**: Doğru tahmin yüzdesi
- **AUC**: Modelin ayırt etme yeteneği
- **F1 Skoru**: Kesinlik ve duyarlılığın dengelenmiş ölçüsü
- **Kesinlik (Precision)**: Spam olarak tahmin edilenlerin ne kadarı gerçekten spam?
- **Duyarlılık (Recall)**: Gerçek spam'lerin ne kadarı yakalandı?

## 📈 Sonuçlar

Model eğitildikten sonra aşağıdaki performans metrikleri elde edilmiştir:

- **Doğruluk (Accuracy)**: %97.28
- **AUC**: 0.9906
- **F1 Skoru**: 0.8997
- **Kesinlik (Precision)**: 0.9630
- **Duyarlılık (Recall)**: 0.8442

## 🎓 Öğrenilen Kavramlar

Bu proje ile aşağıdaki ML.NET ve makine öğrenmesi kavramları öğrenilmiştir:

1. **MLContext**: ML.NET'in ana sınıfı
2. **Veri Yükleme**: CSV'den veri yükleme
3. **Train/Test Split**: Veriyi eğitim ve test setlerine ayırma
4. **Pipeline**: Veri işleme ve model eğitme adımları
5. **Transform**: Veri dönüşümleri (MapValue, FeaturizeText)
6. **Trainer**: Model algoritmaları (SdcaLogisticRegression)
7. **Fit**: Model eğitimi
8. **Transform**: Tahmin yapma
9. **Evaluate**: Model performansını ölçme
10. **Overfitting**: Aşırı öğrenme kavramı
11. **Binary Classification**: İkili sınıflandırma

## 📝 Notlar

- Model her çalıştırmada yeniden eğitilir
- Veri seti Kaggle'dan indirilmiştir
- Proje eğitim amaçlıdır

## 🔗 Kaynaklar

- [ML.NET Dokümantasyonu](https://learn.microsoft.com/dotnet/machine-learning/)
- [Kaggle](https://www.kaggle.com/)

## 👤 Geliştirici

Bu proje ML.NET öğrenmek amacıyla oluşturulmuştur.
