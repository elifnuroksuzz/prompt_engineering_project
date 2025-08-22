from typing import Dict, List, Any
from . import BaseTask, TaskResult
import re

class TextClassificationTask(BaseTask):
    def __init__(self, model_manager, prompt_library, config):
        super().__init__(model_manager, prompt_library, config)
        self.valid_labels = ["Olumlu", "Olumsuz", "Nötr"]
    
    def get_task_name(self) -> str:
        return "Text Classification - Sentiment Analysis"
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Duygu analizi test verilerini döndür"""
        return [
            {
                "input_text": "Bugün hava çok güzel, dışarı çıkmak harika olurdu!",
                "expected_output": "Olumlu"
            },
            {
                "input_text": "Trafik yüzünden işe geç kaldım, çok sinir bozucu.",
                "expected_output": "Olumsuz"
            },
            {
                "input_text": "Toplantı saat 10:00'da başlayacak.",
                "expected_output": "Nötr"
            },
            {
                "input_text": "Bu kitabı gerçekten çok sevdim, herkese tavsiye ederim.",
                "expected_output": "Olumlu"
            },
            {
                "input_text": "Film beklentilerimin altında kaldı, hayal kırıklığına uğradım.",
                "expected_output": "Olumsuz"
            }
        ]
    
    def _generate_prompt(self, strategy: str, data_item: Dict[str, Any]) -> str:
        """Strateji ve veri için prompt oluştur"""
        text = data_item["input_text"]
        
        if strategy == "zero_shot":
            return f"Bu metni 'Olumlu', 'Olumsuz' veya 'Nötr' olarak sınıflandır:\nMetin: '{text}'\nSınıf:"
        
        elif strategy == "one_shot":
            return f"""Aşağıdaki örnekte olduğu gibi metnin duygu durumunu sınıflandır:

Örnek:
Metin: 'Hava harika, güneş parlıyor.'
Duygu: Olumlu

Şimdi sınıflandır:
Metin: '{text}'
Duygu:"""
        
        elif strategy == "few_shot":
            return self.prompt_library.format_prompt(
                "text_classification", 
                "sentiment_few_shot", 
                text=text
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def evaluate_response(self, expected: str, actual: str) -> float:
        """Model yanıtını değerlendir"""
        # Yanıttan sadece sınıf etiketini çıkar
        predicted_label = self._extract_label(actual)
        
        if predicted_label and predicted_label.lower() == expected.lower():
            return 1.0
        return 0.0
    
    def _extract_label(self, response: str) -> str:
        """Model yanıtından sınıf etiketini çıkar"""
        response_lower = response.lower().strip()
        
        # Doğrudan eşleşme kontrolü
        for label in self.valid_labels:
            if label.lower() in response_lower:
                return label
        
        # JSON formatında kontrol
        if "olumlu" in response_lower:
            return "Olumlu"
        elif "olumsuz" in response_lower:
            return "Olumsuz"
        elif "nötr" in response_lower:
            return "Nötr"
        
        return None
    
    def get_accuracy_summary(self) -> Dict[str, float]:
        """Strateji bazında doğruluk özetini döndür"""
        summary = {}
        
        for strategy in ["zero_shot", "one_shot", "few_shot"]:
            strategy_results = [r for r in self.results if r.prompt_type == strategy]
            if strategy_results:
                accuracies = [r.accuracy for r in strategy_results if r.accuracy is not None]
                if accuracies:
                    summary[strategy] = sum(accuracies) / len(accuracies)
        
        return summary