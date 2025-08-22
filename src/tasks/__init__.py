from abc import ABC, abstractmethod
from typing import Dict, List, Any
from dataclasses import dataclass
import pandas as pd


@dataclass
class TaskResult:
    task_name: str
    prompt_type: str
    prompt_format: str
    input_text: str
    model_response: str
    expected_output: str = None
    accuracy: float = None
    metadata: Dict[str, Any] = None

class BaseTask(ABC):
    def __init__(self, model_manager, prompt_library, config):
        self.model_manager = model_manager
        self.prompt_library = prompt_library
        self.config = config
        self.results: List[TaskResult] = []
    
    @abstractmethod
    def get_task_name(self) -> str:
        """Görev adını döndür"""
        pass
    
    @abstractmethod
    def get_test_data(self) -> List[Dict[str, Any]]:
        """Test verilerini döndür"""
        pass
    
    @abstractmethod
    def evaluate_response(self, expected: str, actual: str) -> float:
        """Model yanıtını değerlendir"""
        pass
    
    def run_experiment(self, strategies: List[str] = None) -> pd.DataFrame:
        """Görev deneyimini çalıştır"""
        if strategies is None:
            strategies = ["zero_shot", "one_shot", "few_shot"]
        
        test_data = self.get_test_data()
        
        for strategy in strategies:
            for data_item in test_data:
                self._run_single_test(strategy, data_item)
        
        return self._results_to_dataframe()
    
    def _run_single_test(self, strategy: str, data_item: Dict[str, Any]) -> None:
        """Tek bir test durumunu çalıştır"""
        try:
            prompt = self._generate_prompt(strategy, data_item)
            response = self.model_manager.generate(prompt)
            
            accuracy = None
            if "expected_output" in data_item:
                accuracy = self.evaluate_response(
                    data_item["expected_output"], 
                    response
                )
            
            result = TaskResult(
                task_name=self.get_task_name(),
                prompt_type=strategy,
                prompt_format=self._get_prompt_format_name(strategy),
                input_text=data_item.get("input_text", ""),
                model_response=response,
                expected_output=data_item.get("expected_output"),
                accuracy=accuracy,
                metadata=data_item.get("metadata", {})
            )
            
            self.results.append(result)
            
        except Exception as e:
            print(f"Test failed for {strategy}: {str(e)}")
    
    @abstractmethod
    def _generate_prompt(self, strategy: str, data_item: Dict[str, Any]) -> str:
        """Strateji ve veri için prompt oluştur"""
        pass
    
    def _get_prompt_format_name(self, strategy: str) -> str:
        """Strateji adını format adına çevir"""
        format_names = {
            "zero_shot": "Zero-shot",
            "one_shot": "One-shot", 
            "few_shot": "Few-shot"
        }
        return format_names.get(strategy, strategy)
    
    def _results_to_dataframe(self) -> pd.DataFrame:
        """Sonuçları DataFrame'e çevir"""
        data = []
        for result in self.results:
            data.append({
                "Task": result.task_name,
                "Prompt Type": result.prompt_type,
                "Prompt Format": result.prompt_format,
                "Input": result.input_text,
                "Response": result.model_response,
                "Expected": result.expected_output,
                "Accuracy": result.accuracy
            })
        return pd.DataFrame(data)