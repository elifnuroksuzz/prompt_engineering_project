from typing import Dict, List, Any, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

class EvaluationMetrics:
    def __init__(self):
        self.results = []
    
    def calculate_accuracy(self, expected: List[str], predicted: List[str]) -> float:
        """Basit doğruluk hesaplama"""
        if len(expected) != len(predicted):
            raise ValueError("Expected and predicted lists must have same length")
        
        correct = sum(1 for e, p in zip(expected, predicted) if e == p)
        return correct / len(expected) if expected else 0.0
    
    def generate_classification_report(self, expected: List[str], predicted: List[str], 
                                    labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Detaylı sınıflandırma raporu"""
        try:
            report = classification_report(
                expected, predicted, 
                labels=labels, 
                output_dict=True, 
                zero_division=0
            )
            return report
        except Exception as e:
            print(f"Classification report error: {e}")
            return {}
    
    def calculate_strategy_performance(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Strateji bazında performans analizi"""
        if 'Prompt Type' not in results_df.columns or 'Accuracy' not in results_df.columns:
            return pd.DataFrame()
        
        performance = results_df.groupby(['Prompt Type', 'Prompt Format'])['Accuracy'].agg([
            'mean', 'std', 'count'
        ]).round(3)
        
        return performance.reset_index()