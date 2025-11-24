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
    }
}
