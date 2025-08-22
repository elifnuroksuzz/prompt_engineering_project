# Prompt Engineering Framework

**Gelişmiş AI model performans testi ve prompt stratejisi analiz sistemi**

## Özellikler

- **Çoklu Prompt Stratejileri**: Zero-shot, One-shot, Few-shot, Chain-of-Thought
- **Otomatik Değerlendirme**: Sayısal doğruluk, format analizi, açıklama kalitesi
- **Kapsamlı Raporlama**: CSV, JSON, HTML formatlarında detaylı analiz
- **Real-time API Entegrasyonu**: Gemini AI modeli ile gerçek zamanlı test
- **İstatistiksel Analiz**: Strateji karşılaştırması ve performans metrikleri

## Desteklenen Görevler

### 📝 Text Classification
- Duygu analizi (Sentiment Analysis)
- Türkçe metin sınıflandırma
- Çoklu etiket desteği

### 🧮 Mathematical Reasoning
- İki bilinmeyenli denklem sistemleri
- Adım adım çözüm analizi
- Chain-of-Thought değerlendirmesi

## Kurulum

```bash
git clone https://github.com/yourusername/prompt-engineering-framework
cd prompt-engineering-framework
pip install -r requirements.txt
```

## Konfigürasyon

1. `config/.env` dosyasını oluşturun:
```bash
GEMINI_API_KEY=your_api_key_here
```

2. `config/settings.yaml` dosyasını düzenleyin:
```yaml
model:
  name: "gemini-2.5-flash"
  temperature: 0.1
  mock_mode: false  # Gerçek API için
```

## Kullanım

### Temel Komutlar

```bash
# Mevcut görevleri listele
python main.py --list-tasks

# Tek görev çalıştır
python main.py --task mathematical_reasoning --strategies vanilla zero_shot_cot

# Tüm görevleri çalıştır
python main.py --run-all

# Strateji karşılaştırması
python main.py --task text_classification --compare-strategies

# Performance benchmark
python main.py --benchmark
```

### Gelişmiş Özellikler

```bash
# HTML raporu oluştur
python main.py --task mathematical_reasoning --output-format html

# Özel konfigürasyon
python main.py --task text_classification --config config/test_settings.yaml
```

## Çıktı Örnekleri

### Konsol Çıktısı
```
==================================================
EXPERIMENT RESULTS SUMMARY
==================================================
Performance by Strategy:
  Prompt Type Prompt Format  mean  std  count
      vanilla       vanilla   1.0  0.0      3
zero_shot_cot zero_shot_cot   1.0  0.0      3

Total tests: 6
Average accuracy: 1.000
```

### Dosya Çıktıları
- `data/output/mathematical_reasoning_20250823_001839.csv`
- `data/output/reports/comprehensive_report_20250823_002729.html`
- `data/output/experiment_summary.json`

## Proje Yapısı

```
prompt_engineering_project/
├── config/
│   ├── settings.yaml          # Ana konfigürasyon
│   ├── test_settings.yaml     # Test konfigürasyonu
│   └── .env                   # API anahtarları
├── src/
│   ├── core/
│   │   ├── config.py          # Konfigürasyon yönetimi
│   │   └── model_manager.py   # AI model entegrasyonu
│   ├── tasks/
│   │   ├── text_classification.py
│   │   └── mathematical_reasoning.py
│   ├── prompts/
│   │   └── prompt_library.py  # Prompt şablonları
│   ├── evaluation/
│   │   └── metrics.py         # Değerlendirme metrikleri
│   └── utils/
│       ├── data_handler.py    # Veri işleme
│       └── benchmark_runner.py
├── data/
│   └── output/               # Test sonuçları
├── main.py                   # Ana çalıştırma dosyası
└── requirements.txt
```

## Kavramlar

### Prompt Engineering Stratejileri

- **Zero-shot**: Model'e örnek vermeden direkt görev
- **One-shot**: Tek örnek ile görev
- **Few-shot**: Birkaç örnek ile görev  
- **Chain-of-Thought (CoT)**: Adım adım düşünme süreci

### Değerlendirme Metrikleri

- **Numeric Accuracy**: Sayısal sonuçların doğruluğu
- **Format Correctness**: Yanıt formatının uygunluğu
- **Explanation Quality**: Açıklama kalitesi analizi

### Performance Analizi

- **Strategy Comparison**: Strateji bazında performans
- **Statistical Analysis**: Ortalama, standart sapma, tutarlılık
- **Benchmark Testing**: Hız ve bellek kullanımı

## API Entegrasyonu

Framework şu AI modelleri destekler:
- Google Gemini (varsayılan)
- Mock mode (test için)

## Geliştirme

### Yeni Görev Ekleme

```python
class NewTask(BaseTask):
    def get_task_name(self) -> str:
        return "Your Task Name"
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        return [{"input_text": "...", "expected_output": "..."}]
    
    def evaluate_response(self, expected: str, actual: str) -> float:
        # Değerlendirme mantığı
        return score
```

### Yeni Prompt Stratejisi

```python
def _generate_prompt(self, strategy: str, data_item: Dict[str, Any]) -> str:
    if strategy == "your_strategy":
        return f"Your prompt template: {data_item['input_text']}"
```

## Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Branch'i push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## Lisans

MIT License - Detaylar için `LICENSE` dosyasına bakın.

## İletişim

- **Proje**: [GitHub Repository](https://github.com/yourusername/prompt-engineering-framework)
- **Issues**: Bug raporları ve özellik istekleri için GitHub Issues kullanın

---

**Not**: Bu framework akademik araştırma ve prompt engineering optimizasyonu için tasarlanmıştır. Production kullanımı için ek güvenlik ve ölçeklendirme önlemleri gerekebilir.
