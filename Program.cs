using Microsoft.ML;

namespace SpamClassifier;

class Program
{
    static void Main(string[] args)
    {
        // ML.NET'in ana sınıfı: MLContext
        // Bu sınıf, tüm ML.NET işlemlerinin başlangıç noktasıdır
        // Seed: 0 → Rastgele sayı üretimi için başlangıç değeri (tekrarlanabilirlik için)
        var mlContext = new MLContext(seed: 0);

        Console.WriteLine("ML.NET Spam Sınıflandırıcı - Başlatılıyor...");
        Console.WriteLine("MLContext oluşturuldu!");

        // Adım 1: CSV dosyasından veriyi yükle
        // LoadFromTextFile: CSV dosyasını ML.NET'in IDataView formatına dönüştürür
        // separatorChar: ',' → CSV'deki ayırıcı karakter
        // hasHeader: true → İlk satır başlık satırı (v1, v2, ...)
        // DataPath: CSV dosyasının yolu (Data klasörü içinde)
        var dataPath = Path.Combine("Data", "spam.csv");

        Console.WriteLine($"Veri dosyası yükleniyor: {dataPath}");

        // CSV'deki sütun isimleri (v1, v2) ile SpamData sınıfındaki property'leri eşleştir
        // ML.NET otomatik olarak eşleştirme yapar, ama biz manuel olarak belirtebiliriz
        var dataView = mlContext.Data.LoadFromTextFile<SpamData>(
            path: dataPath,
            separatorChar: ',',
            hasHeader: true);

        // Kaç satır veri yüklendi?
        // GetRowCount() bazen null döndürebilir, bu yüzden veriyi sayıyoruz
        // CreateEnumerable: IDataView'ı IEnumerable'a dönüştürür (veriyi okumak için)
        // reuseRowObject: false → Her satır için yeni obje oluştur (performans için)
        var rowCount = mlContext.Data.CreateEnumerable<SpamData>(dataView, reuseRowObject: false).Count();

        Console.WriteLine($"✓ {rowCount} satır veri yüklendi!");

        // Adım 2: Veriyi Eğitim ve Test Setlerine Ayırma
        // TrainTestSplit: Veriyi rastgele olarak iki parçaya böler
        // testFraction: 0.2 → %20 test, %80 eğitim
        // seed: 0 → Rastgele bölme için başlangıç değeri (tekrarlanabilirlik için)
        Console.WriteLine("\nVeriyi eğitim ve test setlerine ayırılıyor...");
        var dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 0);

        var trainData = dataSplit.TrainSet;  // Eğitim verisi (%80)
        var testData = dataSplit.TestSet;     // Test verisi (%20)

        var trainCount = mlContext.Data.CreateEnumerable<SpamData>(trainData, reuseRowObject: false).Count(); //egitim verisi sayisi
        var testCount = mlContext.Data.CreateEnumerable<SpamData>(testData, reuseRowObject: false).Count(); //test verisi sayisi

        Console.WriteLine($"✓ Eğitim seti: {trainCount} satır");
        Console.WriteLine($"✓ Test seti: {testCount} satır");

        // Adım 3: ML.NET Pipeline Oluşturma
        // Pipeline: Veriyi işleme ve model eğitme adımlarının sırası
        Console.WriteLine("\nPipeline oluşturuluyor...");

        // Pipeline oluştur: Veriyi işleme adımlarını tanımla
        // BinaryClassification trainer'lar Boolean label bekliyor, bu yüzden string'i Boolean'a dönüştürmeliyiz
        var pipeline = mlContext.Transforms.Conversion.MapValue(
                outputColumnName: "LabelBool",     // Çıktı sütunu: Boolean label
                inputColumnName: "Label",          // Giriş sütunu: "ham" veya "spam" string'i
                keyValuePairs: new[]
                {
                    new KeyValuePair<string, bool>("ham", false),   // "ham" → false
                    new KeyValuePair<string, bool>("spam", true)   // "spam" → true
                })
            // MapValue: String label'ları Boolean değerlere dönüştürür
            // "ham" → false (Boolean)
            // "spam" → true (Boolean)
            // BinaryClassification trainer'lar Boolean label bekliyor!

            .Append(mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",      // Çıktı sütunu: Sayısal özellikler
                inputColumnName: "Message"))       // Giriş sütunu: Mesaj metni
                                                   // FeaturizeText: Metni sayısal özelliklere dönüştürür
                                                   // Örnek: "Free money!" → [0.5, 0.8, 0.2, ...] (sayı dizisi)
                                                   // Neden? Makine öğrenmesi algoritmaları sayılarla çalışır, metinle değil!

            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "LabelBool",      // Hangi sütun tahmin edilecek? (Boolean label)
                featureColumnName: "Features"));   // Hangi sütun özellikler?
                                                   // SdcaLogisticRegression: Binary (ikili) sınıflandırma algoritması
                                                   // Binary Classification: İki sınıf (ham/spam) arasında karar verir
                                                   // Logistic Regression: Olasılık tabanlı sınıflandırma yapar
                                                   // SDCA: Stochastic Dual Coordinate Ascent (optimizasyon yöntemi)

        Console.WriteLine("✓ Pipeline oluşturuldu!");

        // Adım 4: Model Eğitimi
        // Fit: Pipeline'ı eğitim verisiyle çalıştırır ve modeli eğitir
        // Model, eğitim verisinden öğrenir ve tahmin yapmayı öğrenir
        Console.WriteLine("\nModel eğitiliyor... (Bu biraz zaman alabilir)");

        var model = pipeline.Fit(trainData);
        // Fit metodu:
        // 1. Eğitim verisini alır
        // 2. Pipeline'daki tüm dönüşümleri uygular (FeaturizeText, vb.)
        // 3. Algoritmayı (SdcaLogisticRegression) eğitir
        // 4. Eğitilmiş modeli döndürür

        Console.WriteLine("✓ Model eğitimi tamamlandı!");

        // Adım 5: Model Değerlendirme
        // Test verisiyle modeli test ediyoruz
        // Modelin ne kadar iyi çalıştığını ölçüyoruz
        Console.WriteLine("\nModel test ediliyor...");

        // Test verisini dönüştür (pipeline'ı uygula)
        var predictions = model.Transform(testData); //eğitim verisindn öğrendiği ile test verisinde tahmin yapar
        // Transform: Eğitilmiş modeli test verisine uygular
        // Test verisindeki her mesaj için tahmin yapar

        // Model performansını ölç
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: "LabelBool");
        // Evaluate: Modelin ne kadar iyi çalıştığını ölçer
        // BinaryClassification.Evaluate: İkili sınıflandırma için metrikler hesaplar

        // Sonuçları göster
        Console.WriteLine("\n=== Model Performans Metrikleri ===");
        Console.WriteLine($"Doğruluk (Accuracy): {metrics.Accuracy:P2}");
        // Doğruluk: Doğru tahmin yüzdesi
        // Örnek: 0.95 = %95 doğru tahmin

        Console.WriteLine($"AUC (Eğri Altındaki Alan): {metrics.AreaUnderRocCurve:F4}");
        // AUC: Modelin ayırt etme yeteneği (0-1 arası, 1'e yakın = daha iyi)
        // ROC Eğrisi: Doğru Pozitif Oranı vs Yanlış Pozitif Oranı grafiği

        Console.WriteLine($"F1 Skoru: {metrics.F1Score:F4}");
        // F1 Skoru: Kesinlik ve Duyarlılığın harmonik ortalaması (0-1 arası, 1'e yakın = daha iyi)

        Console.WriteLine($"Kesinlik (Precision): {metrics.PositivePrecision:F4}");
        // Kesinlik: Spam olarak tahmin edilenlerin ne kadarı gerçekten spam?

        Console.WriteLine($"Duyarlılık (Recall): {metrics.PositiveRecall:F4}");
        // Duyarlılık: Gerçek spam'lerin ne kadarı yakalandı?
    }
}
