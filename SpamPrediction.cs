using Microsoft.ML.Data;

namespace SpamClassifier;

/// <summary>
/// Modelin tahmin sonucu
/// </summary>
public class SpamPrediction
{
    /// <summary>
    /// Tahmin edilen etiket: "ham" veya "spam"
    /// </summary>
    [ColumnName("PredictedLabel")]
    public string PredictedLabel { get; set; } = string.Empty;

    /// <summary>
    /// Tahmin skorları: Her sınıf için olasılık değerleri
    /// Sınıflar: [0] = "ham" olasılığı, [1] = "spam" olasılığı
    /// Örnek: Score = [0.92, 0.08] → %92 ham, %8 spam
    /// </summary>
    public float[] Score { get; set; } = Array.Empty<float>();
}

