import re
from typing import List, Dict, Any
import pandas as pd
from . import BaseTask, TaskResult

class MathematicalReasoningTask(BaseTask):
    def __init__(self, model_manager, prompt_library, config):
        super().__init__(model_manager, prompt_library, config)
    
    def get_task_name(self) -> str:
        return "Mathematical Reasoning - CoT"
    
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Matematiksel problem test verilerini döndür"""
        return [
            {
                "input_text": "Bir manavda, 5 kilo elma ve 3 kilo portakal alan bir müşteri toplam 42 TL ödüyor. Eğer 1 kilo elma, 1 kilo portakaldan 2 TL daha pahalı ise, 1 kilo elma ve 1 kilo portakalın fiyatı ayrı ayrı kaç TL'dir?",
                "expected_output": "1 kilo elma: 6 TL, 1 kilo portakal: 4 TL",
                "expected_numbers": [6, 4],
                "problem_type": "equation_system"
            },
            {
                "input_text": "Bir kafede 4 kahve ve 2 çay 26 TL tutuyor. 1 kahve, 1 çaydan 3 TL daha pahalı ise, 1 kahve ve 1 çayın fiyatı nedir?",
                "expected_output": "1 kahve: 5 TL, 1 çay: 2 TL",
                "expected_numbers": [5, 2],
                "problem_type": "equation_system"
            },
            {
                "input_text": "Bir parkta 6 çocuk ve 4 yetişkin için bilet toplam 84 TL. Çocuk bileti, yetişkin biletinden 6 TL daha ucuz ise, bilet fiyatları nedir?",
                "expected_output": "Çocuk bileti: 6 TL, Yetişkin bileti: 12 TL",
                "expected_numbers": [6, 12],
                "problem_type": "equation_system"
            }
        ]
    
    def _generate_prompt(self, strategy: str, data_item: Dict[str, Any]) -> str:
        """Strateji ve veri için prompt oluştur"""
        problem = data_item["input_text"]
        
        if strategy == "vanilla":
            return f"Aşağıdaki problemi çözün:\n{problem}"
        
        elif strategy == "zero_shot_cot":
            return f"Aşağıdaki problemi çözün. Adım adım düşünelim.\n\n{problem}"
        
        elif strategy == "few_shot_cot":
            return self.prompt_library.format_prompt(
                "mathematical_reasoning", 
                "equation_systems_few_shot", 
                problem=problem
            )
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def evaluate_response(self, expected: str, actual: str) -> float:
        """Geliştirilmiş değerlendirme sistemi"""
        # Sayısal değerleri çıkar
        expected_numbers = self._extract_numbers(expected)
        actual_numbers = self._extract_numbers(actual)
        
        print(f"Expected: {expected}")
        print(f"Actual: {actual[:200]}...")
        print(f"Expected numbers: {expected_numbers}")
        print(f"Actual numbers: {actual_numbers}")
        
        # Çoklu değerlendirme kriterleri
        scores = []
        
        # 1. Sayısal doğruluk
        numeric_score = self._evaluate_numeric_accuracy(expected_numbers, actual_numbers)
        scores.append(("numeric", numeric_score, 0.6))  # %60 ağırlık
        
        # 2. Format doğruluğu
        format_score = self._evaluate_format_correctness(actual)
        scores.append(("format", format_score, 0.2))  # %20 ağırlık
        
        # 3. Açıklama kalitesi
        explanation_score = self._evaluate_explanation_quality(actual)
        scores.append(("explanation", explanation_score, 0.2))  # %20 ağırlık
        
        # Ağırlıklı ortalama
        total_score = sum(score * weight for _, score, weight in scores)
        
        print(f"Scoring breakdown: {[(name, f'{score:.2f}') for name, score, _ in scores]}")
        print(f"Final score: {total_score:.2f}")
        
        return total_score
    
    def _evaluate_numeric_accuracy(self, expected_numbers: List[int], actual_numbers: List[int]) -> float:
        """Sayısal doğruluk değerlendirmesi"""
        if not expected_numbers:
            return 0.0
        
        # Tam eşleşme kontrolü
        if set(expected_numbers).issubset(set(actual_numbers)):
            return 1.0
        
        # Kısmi eşleşme
        common_numbers = set(expected_numbers) & set(actual_numbers)
        if len(common_numbers) >= len(expected_numbers) // 2:
            return 0.7
        elif len(common_numbers) > 0:
            return 0.4
        
        return 0.0
    
    def _evaluate_format_correctness(self, response: str) -> float:
        """Format doğruluğu değerlendirmesi"""
        response_lower = response.lower()
        score = 0.0
        
        # TL/Lira formatı
        if "tl" in response_lower or "lira" in response_lower:
            score += 0.3
        
        # Fiyat belirtimi
        if any(word in response_lower for word in ["fiyat", "bilet", "elma", "portakal", "kahve", "çay"]):
            score += 0.4
        
        # Sayısal format (X TL şeklinde)
        if re.search(r'\d+\s*tl', response_lower):
            score += 0.3
        
        return min(score, 1.0)
    
    def _evaluate_explanation_quality(self, response: str) -> float:
        """Açıklama kalitesi değerlendirmesi"""
        response_lower = response.lower()
        score = 0.0
        
        # Adım adım çözüm
        if any(word in response_lower for word in ["adım", "önce", "sonra", "denklem"]):
            score += 0.4
        
        # Matematiksel terimler
        math_terms = ["bilinmeyen", "değişken", "çöz", "hesap", "toplam"]
        if any(term in response_lower for term in math_terms):
            score += 0.3
        
        # Yapılandırılmış sunum
        if response.count('\n') > 3:  # Çok satırlı açıklama
            score += 0.3
        
        return min(score, 1.0)
    
    def _extract_numbers(self, text: str) -> List[int]:
        """Metinden sayıları çıkar"""
        numbers = re.findall(r'\b\d+\b', text)
        return [int(n) for n in numbers]
    
    def run_experiment(self, strategies: List[str] = None) -> pd.DataFrame:
        """CoT için özel strateji listesi"""
        if strategies is None:
            strategies = ["vanilla", "zero_shot_cot", "few_shot_cot"]
        
        return super().run_experiment(strategies)
    
    def get_detailed_analysis(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Detaylı analiz raporu"""
        analysis = {
            "overview": {
                "total_tests": len(results_df),
                "average_accuracy": results_df['Accuracy'].mean(),
                "perfect_scores": (results_df['Accuracy'] == 1.0).sum()
            },
            "strategy_performance": {},
            "recommendations": []
        }
        
        # Strateji bazında analiz
        for strategy in results_df['Prompt Type'].unique():
            strategy_data = results_df[results_df['Prompt Type'] == strategy]
            analysis["strategy_performance"][strategy] = {
                "accuracy": strategy_data['Accuracy'].mean(),
                "consistency": 1 - strategy_data['Accuracy'].std(),  # Düşük std = yüksek tutarlılık
                "test_count": len(strategy_data)
            }
        
        # Öneriler
        best_strategy = max(analysis["strategy_performance"].items(), 
                          key=lambda x: x[1]["accuracy"])[0]
        analysis["recommendations"].append(f"En iyi performans: {best_strategy}")
        
        if analysis["overview"]["average_accuracy"] < 0.8:
            analysis["recommendations"].append("Prompt'ların iyileştirilmesi önerilir")
        
        return analysis