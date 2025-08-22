from typing import Dict, List, Any
import json
from dataclasses import dataclass

@dataclass
class PromptTemplate:
    name: str
    description: str
    strategy: str
    template: str
    expected_output: str
    effectiveness_note: str
    use_case: str

class PromptLibrary:
    def __init__(self):
        self._templates: Dict[str, Dict[str, PromptTemplate]] = {}
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """Varsayılan prompt şablonlarını yükle"""
        
        # Text Classification Templates
        self._templates["text_classification"] = {
            "sentiment_few_shot": PromptTemplate(
                name="Duygu Analizi (Few-shot)",
                description="Müşteri yorumlarının duygu sınıflandırması için kullanılır",
                strategy="Few-shot Learning",
                template="""Yorum: "Bu ürün beklentimin altındaydı, hayal kırıklığına uğradım."
Duygu: Olumsuz

Yorum: "Harika bir alışveriş deneyimiydi, herkese tavsiye ederim!"
Duygu: Olumlu

Yorum: "{text}"
Duygu:""",
                expected_output="Tek kelime (Olumlu/Olumsuz/Nötr)",
                effectiveness_note="Genellikle %90+ doğruluk oranı sağlar",
                use_case="Büyük ölçekli duygu analizi"
            )
        }
        
        # Information Extraction Templates  
        self._templates["information_extraction"] = {
            "entity_extraction_structured": PromptTemplate(
                name="Varlık Çıkarma (Yapılandırılmış)",
                description="Metinlerden kişi, yer, tarih bilgilerini çıkarır",
                strategy="Few-shot Learning",
                template="""Metin: "Mustafa Kemal Atatürk, 1881'de Selanik'te doğdu ve Türkiye Cumhuriyeti'nin kurucusudur."
Kişiler: Mustafa Kemal Atatürk
Yerleşim Yerleri: Selanik, Türkiye
Tarihler: 1881

Metin: "{text}"
Kişiler:
Yerleşim Yerleri:
Tarihler:""",
                expected_output="Yapılandırılmış liste formatı",
                effectiveness_note="Format tutarlılığı yüksek",
                use_case="Belge analizi, veri çıkarma"
            )
        }
        
        # Mathematical Reasoning Templates
        self._templates["mathematical_reasoning"] = {
            "equation_systems_few_shot": PromptTemplate(
                name="Denklem Sistemleri (Few-shot CoT)",
                description="İki bilinmeyenli denklem sistemlerini adım adım çözer",
                strategy="Few-shot Chain-of-Thought",
                template="""Problem: Bir kasapta, 3 kilo kıyma ve 2 kilo sucuk alan bir müşteri toplam 90 TL ödüyor. Eğer 1 kilo kıyma, 1 kilo sucuktan 5 TL daha pahalı ise, 1 kilo kıyma ve 1 kilo sucuk fiyatı ayrı ayrı kaç TL'dir?

Çözüm:
1. Kıyma fiyatına 'k', sucuk fiyatına 's' diyelim.
2. Problemin ilk cümlesinden denklemi kuralım: 3k + 2s = 90
3. Problemin ikinci cümlesinden denklemi kuralım: k = s + 5
4. İkinci denklemi ilk denklemde yerine koyalım: 3(s + 5) + 2s = 90
5. Denklemi açalım: 3s + 15 + 2s = 90
6. Benzer terimleri birleştirelim: 5s + 15 = 90
7. 15'i diğer tarafa atalım: 5s = 90 - 15
8. İşlemi yapalım: 5s = 75
9. s'yi bulalım: s = 75 / 5 = 15
10. Sucuğun fiyatı 15 TL'dir.
11. Kıyma fiyatını bulmak için k = s + 5 denklemini kullanalım: k = 15 + 5 = 20
12. Kıymanın fiyatı 20 TL'dir.

Yanıt: 1 kilo kıyma: 20 TL, 1 kilo sucuk: 15 TL

Problem: {problem}
Çözüm:""",
                expected_output="Adım adım çözüm ve nihai yanıt",
                effectiveness_note="Karmaşık problemlerde muhakeme yeteneğini artırır",
                use_case="Matematiksel soru çözümleri"
            )
        }
    
    def get_template(self, category: str, template_name: str) -> PromptTemplate:
        """Belirli bir template'i getir"""
        if category in self._templates and template_name in self._templates[category]:
            return self._templates[category][template_name]
        raise KeyError(f"Template not found: {category}.{template_name}")
    
    def get_category_templates(self, category: str) -> Dict[str, PromptTemplate]:
        """Kategori altındaki tüm template'leri getir"""
        return self._templates.get(category, {})
    
    def format_prompt(self, category: str, template_name: str, **kwargs) -> str:
        """Template'i parametrelerle formatla"""
        template = self.get_template(category, template_name)
        return template.template.format(**kwargs)