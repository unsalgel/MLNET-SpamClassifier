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

        Console.WriteLine("ML.NET Spam Classifier - Başlatılıyor...");
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
        var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
                outputColumnName: "LabelKey",      // Çıktı sütunu: Sayısal label
                inputColumnName: "Label")          // Giriş sütunu: "ham" veya "spam" string'i
                                                   // MapValueToKey: String label'ları sayısal değerlere dönüştürür
                                                   // Örnek: "ham" → 0, "spam" → 1
                                                   // ML.NET algoritmaları sayısal değerlerle çalışır, string'lerle değil!

            .Append(mlContext.Transforms.Text.FeaturizeText(
                outputColumnName: "Features",      // Çıktı sütunu: Sayısal özellikler
                inputColumnName: "Message"))       // Giriş sütunu: Mesaj metni
                                                   // FeaturizeText: Metni sayısal özelliklere dönüştürür
                                                   // Örnek: "Free money!" → [0.5, 0.8, 0.2, ...] (sayı dizisi)
                                                   // Neden? Makine öğrenmesi algoritmaları sayılarla çalışır, metinle değil!

            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: "LabelKey",       // Hangi sütun tahmin edilecek?
                featureColumnName: "Features"))    // Hangi sütun özellikler?
                                                   // SdcaLogisticRegression: Binary (ikili) sınıflandırma algoritması
                                                   // Binary Classification: İki sınıf (ham/spam) arasında karar verir
                                                   // Logistic Regression: Olasılık tabanlı sınıflandırma yapar
                                                   // SDCA: Stochastic Dual Coordinate Ascent (optimizasyon yöntemi)

            .Append(mlContext.Transforms.Conversion.MapKeyToValue(
                outputColumnName: "PredictedLabel",
                inputColumnName: "PredictedLabel"));
        // MapKeyToValue: Sayısal tahmini tekrar string'e dönüştürür
        // Örnek: 0 → "ham", 1 → "spam"
        // Böylece sonuç okunabilir olur!

        Console.WriteLine("✓ Pipeline oluşturuldu!");
    }
}
