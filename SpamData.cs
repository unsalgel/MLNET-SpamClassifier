using Microsoft.ML.Data;

namespace SpamClassifier;

/// <summary>
/// CSV dosyasından okunan ham veri modeli
/// </summary>
public class SpamData
{
    /// <summary>
    /// Etiket: "ham" veya "spam"
    /// LoadColumn(0): CSV'deki ilk sütun (v1) bu property'ye yüklenecek
    /// </summary>
    [LoadColumn(0)]
    public string Label { get; set; } = string.Empty;

    /// <summary>
    /// E-posta mesajının içeriği
    /// LoadColumn(1): CSV'deki ikinci sütun (v2) bu property'ye yüklenecek
    /// </summary>
    [LoadColumn(1)]
    public string Message { get; set; } = string.Empty;
}

